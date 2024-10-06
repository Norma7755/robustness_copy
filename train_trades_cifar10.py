import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms
from autoattack import AutoAttack
from utils.args import Build_Parser
from utils.evalution import eval_test, adv_test, adv_eval_train, eval_train
from utils.cifar10 import build_model
from utils.trades import train

# CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --AT_type TRADE --model_name SSM --attack_type PGD
# CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --AT_type Madry --model_name SSM --attack_type PGD
# CUDA_VISIBLE_DEVICES=2 python train_trades_cifar10.py --AT_type Nat --model_name SSM --attack_type PGD

# srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type Madry --model_name Mega --attack_type PGD 
# srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type TRADE --model_name Mega --attack_type PGD 
#srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type Madry --model_name S6 --attack_type PGD
#srun -p llmit6 --pty --cpus-per-task=12 --gres=gpu:1 --mem-per-cpu 16384 python train_trades_cifar10.py --AT_type TRADE --model_name S6 --attack_type PGD

# python train_trades_cifar10.py --AT_type Madry --model_name Res18 --attack_type PGD --num_layers 6
# python train_trades_cifar10.py --AT_type TRADE --model_name Res18 --attack_type PGD --num_layers 6

# CUDA_VISIBLE_DEVICES=0 python train_trades_cifar10.py --AT_type Madry --model_name Res18 --attack_type PGD --num_layers 6
# CUDA_VISIBLE_DEVICES=1 python train_trades_cifar10.py --AT_type TRADE --model_name Res18 --attack_type PGD --num_layers 6
# CUDA_VISIBLE_DEVICES=2 python train_trades_cifar10.py --AT_type Madry --model_name SSM --attack_type PGD --num_layers 6
# CUDA_VISIBLE_DEVICES=3 python train_trades_cifar10.py --AT_type TRADE --model_name SSM --attack_type PGD --num_layers 6


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

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers = 12, pin_memory = True)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers = 12, pin_memory = True)



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

def AA_adv_test(model,args,log_path):
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

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            

def main():
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
    for epoch in range(1, args.epochs + 1):
        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch, args.AT_type)

        # evaluation on natural examples
        # print('================================================================')
        # train_loss, train_acc = eval_train(args, model, device, train_loader)
        
        # test_loss, test_acc = eval_test(args, model, device, test_loader)
        # adv_test_loss, adv_test_acc = adv_test(args, model, device, test_loader, args.attack_type)

        # if args.AT_type != 'Nat':
        #     adv_test_loss, adv_test_acc = AA_adv_test()
            # adv_train_loss, adv_train_acc = adv_eval_train(args, model, device, train_loader, args.attack_type)
        print('================================================================')
        if args.AT_type != 'Nat' and epoch % args.AA_lags == 0:
            with open(log_path, 'a') as f:
                f.write('now epoch:' + str(epoch) + '\n')
                f.flush()
            AA_adv_test(model, args, log_path)
            # log_file.write('Epoch {}, Adv Test Loss: {:.4f}, Adv Test Acc: {:.4f}\n'
            #      .format(epoch, adv_test_loss, adv_test_acc))
            # log_file.write('Epoch {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Adv Train Loss: {:.4f}, Adv Train Acc: {:.4f}, Adv Test Loss: {:.4f}, Adv Test Acc: {:.4f}\n'
            #      .format(epoch, train_loss, train_acc, test_loss, test_acc, adv_train_loss, adv_train_acc, adv_test_loss, adv_test_acc))
        # else:
        #     log_file.write('Epoch {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Adv Test Loss: {:.4f}, Adv Test Acc: {:.4f}\n'
        #          .format(epoch, train_loss, train_acc, test_loss, test_acc, adv_test_loss, adv_test_acc))
    
        # save checkpoint
        if epoch % args.save_freq == 0:
            if args.use_inject:
                torch.save(model.state_dict(),
                       os.path.join(model_dir, args.model_name+args.AT_type+'{}'.format(args.inject_method)+ '-epoch{}.pt'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                       os.path.join(model_dir, args.model_name+args.AT_type+ '-epoch{}.pt'.format(epoch)))
        scheduler.step()

if __name__ == '__main__':
    print(args)
    main()
    # main_eval()