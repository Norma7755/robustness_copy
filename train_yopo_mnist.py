import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from autoattack import AutoAttack
from utils.args import Build_Parser
from utils.YOPO import Hamiltonian, FastGradientLayerOneTrainer, adv_train
from utils.evalution import eval_test, adv_test, adv_eval_train, eval_train
from utils.mnist import build_model
from models.SSM import SSM, SSM_Individual_Head, Mega, S5_SSM, S6_SSM, Transformer


# args
parser = Build_Parser()
args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.ToTensor()),
                   batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

def build_model(args, model_name):
    if model_name == 'SSM':
        if args.rest_lyap:
            model = SSM(d_input=1, d_model=128, n_layers=args.num_layers, use_lyap=True)
        else:
            if args.use_inject:
                model = SSM(d_input=1, d_model=128, n_layers=args.num_layers, \
                    use_inject=True,inject_method=args.inject_method, patch_size=args.patch_size)
            else:
                model = SSM(d_input=1, d_model=128, n_layers=args.num_layers, patch_size=args.patch_size)
    elif model_name == 'DSS':
        if args.use_inject:
            model = SSM(d_input=1, d_model=128, n_layers=args.num_layers, mode = 'diag', \
                use_inject=True,inject_method=args.inject_method, patch_size=args.patch_size)
        else:
            model = SSM(d_input=1, d_model=128, n_layers=args.num_layers, mode = 'diag', patch_size=args.patch_size)
    elif model_name == 'S5':
        model = S5_SSM(d_input=1, d_model=128, n_layers=args.num_layers, patch_size=args.patch_size)
    elif model_name == 'Mega':
        model = Mega(d_input=1, d_model=128, n_layers=args.num_layers, seq_len=28*28, patch_size=args.patch_size) 
    elif model_name == 'S6':
        model = S6_SSM(d_input=1, d_model=128, n_layers=args.num_layers, patch_size=args.patch_size)     
    elif model_name == 'SSM_ind_head':
        model = SSM_Individual_Head(d_input=1)
    elif model_name == 'Transformer':
        model = Transformer(d_input=1, n_layers=args.num_layers, patch_size=args.patch_size)
    return model

def main_eval():
    model = build_model(args, args.model_name).to(device)
    state = torch.load('/root/bqqi/robustness/TRADES/checkpoints/model-cifar/SSMTRADE-epoch180.pt')
    model.load_state_dict(state, strict=False)
    print('================================================================')
    eval_test(args, model, device, test_loader)
    
    adv_test(args, model, device, test_loader, args.attack_type)
    # print(model.avg_vol)
    adv_train_loss, adv_train_acc = adv_eval_train(args, model, device, train_loader, args.attack_type)
    
    print('================================================================')

def AA_eval(model,args):
    model.eval()
    save_dir=os.path.join(args.model_dir,'AA_results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_path = os.path.join(save_dir,'log_file.txt')
    model.eval()
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=log_path,
        version=args.version,verbose=args.AA_verbose)
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    torch.cuda.empty_cache()
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.AA_bs, state_path=args.state_path)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            
            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))
    
def main():
    # init model, ResNet18() can be also used here for training
    if args.use_inject:
        log_path = os.path.join(model_dir,'train_log'+args.model_name+'_inject_{}'.format(args.inject_method)+args.AT_type+'.txt')
    else:    
        log_path = os.path.join(model_dir,'train_log'+args.model_name+args.AT_type+'.txt')

    log_file = open(log_path, 'w')
    model = build_model(args, args.model_name).to(device)
    if args.rest_lyap:
        state = torch.load('/root/bqqi/fscil/robustness/TRADES/checkpoints/model-cifar/SSMTRADE-epoch180_not_trainable.pt')
        model.load_state_dict(state, strict=False)
        optimizer = optim.Adam(model.adjusts.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    Hamiltonian_func = Hamiltonian(model.input, args.weight_decay)
    layer_one_optimizer = optim.AdamW(model.parameters(), lr = scheduler.get_lr()[0], weight_decay=args.weight_decay)
    LayerOneTrainer = FastGradientLayerOneTrainer(Hamiltonian_func, layer_one_optimizer,
                                              3, args.step_size, args.epsilon)

    for epoch in range(1, args.epochs + 1):
        # adversarial training
        adv_train(args, model, device, train_loader, optimizer, epoch, LayerOneTrainer)

        # evaluation on natural examples
        print('================================================================')
        train_loss, train_acc = eval_train(args, model, device, train_loader)
        
        test_loss, test_acc = eval_test(args, model, device, test_loader)
        adv_test_loss, adv_test_acc = adv_test(args, model, device, test_loader, args.attack_type)
        if args.AT_type != 'Nat':
            adv_train_loss, adv_train_acc = adv_eval_train(args, model, device, train_loader, args.attack_type)
        print('================================================================')
        if args.AT_type != 'Nat':
            log_file.write('Epoch {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Adv Train Loss: {:.4f}, Adv Train Acc: {:.4f}, Adv Test Loss: {:.4f}, Adv Test Acc: {:.4f}\n'
                 .format(epoch, train_loss, train_acc, test_loss, test_acc, adv_train_loss, adv_train_acc, adv_test_loss, adv_test_acc))
        else:
            log_file.write('Epoch {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Adv Test Loss: {:.4f}, Adv Test Acc: {:.4f}\n'
                 .format(epoch, train_loss, train_acc, test_loss, test_acc, adv_test_loss, adv_test_acc))
    
        # save checkpoint
        if epoch % args.save_freq == 0:
            if args.use_inject:
                torch.save(model.state_dict(),
                       os.path.join(model_dir, args.model_name+args.AT_type+'{}'.format(args.inject_method)+ '-epoch{}.pt'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                       os.path.join(model_dir, args.model_name+args.AT_type+ '-epoch{}.pt'.format(epoch)))
        scheduler.step()
    AA_eval(model,args)

if __name__ == '__main__':
    print(args)
    main()
    # main_eval()