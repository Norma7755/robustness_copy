import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from autoattack import AutoAttack
from utils.args import Build_Parser
from utils.evalution import eval_test, adv_test, adv_eval_train, eval_train
from utils.mnist import build_model
from utils.trades import train

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
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.ToTensor()),
                   batch_size=args.test_batch_size, shuffle=False, **kwargs)

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
