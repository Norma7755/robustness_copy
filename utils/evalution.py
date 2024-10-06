import torch
import torch.nn.functional as F
from utils.used_attacks import PGD, Lyap_Control
from autoattack.autopgd_base import APGDAttack as APGD

def eval_train(args, model, device, train_loader):
    model.eval()
    
    if args.individual_heads:
        train_loss = [0.0] * args.num_layers
        correct = [0] * args.num_layers
    else:
        train_loss = 0
        correct = 0
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            if args.rest_lyap:
                output, loss_lyap = model(data, ret_lyap = True)
            else:
                output = model(data)
            if args.individual_heads:
                for i, x_out in enumerate(output):
                    train_loss[i] += F.cross_entropy(x_out, target, size_average=False).item()
                    pred = output[i].max(1, keepdim=True)[1]
                    correct[i] += pred.eq(target.view_as(pred)).sum().item()
            else:
                if args.rest_lyap:
                    train_loss += loss_lyap.mean()
                else:
                    train_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                
    train_loss /= len(train_loader.dataset)
    if args.individual_heads:
        correct = correct / len(train_loader.dataset)
        train_loss_str = ', '.join('{:.4f}'.format(loss) for loss in train_loss)
        correct_str = ', '.join(str(corr) for corr in correct)
        print('Train: Average loss: {}, Accuracy: {}%'.format(
        train_loss_str, 100. * correct_str))
        training_accuracy = correct
    else:
        print('Train: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
        training_accuracy = correct / len(train_loader.dataset)
    
    return train_loss, training_accuracy

def adv_eval_train(args, model, device, train_loader, attack_method = 'PGD'):
    model.eval()
    
    if args.individual_heads:
        train_loss = [0.0] * args.num_layers
        correct = [0] * args.num_layers
    else:
        train_loss = 0
        correct = 0
    
    if attack_method == 'PGD':
        attack = PGD(model, args.epsilon, args.step_size, args.num_steps, normalize=False)
    elif args.attack_type =='APGD':
        attack = APGD(model, eps=args.epsilon, seed=args.seed,device=device)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        if args.attack_type =='APGD':
            data_adv = attack.perturb(data, target)
        else:
            data_adv = attack(data, target)
        with torch.no_grad():
            if args.rest_lyap:
                output, loss_lyap = model(data_adv, ret_lyap = True)
            else:
                output = model(data_adv)
                    
            if args.individual_heads:
                for i, x_out in enumerate(output):
                    train_loss[i] += F.cross_entropy(x_out, target, size_average=False).item()
                    pred = output[i].max(1, keepdim=True)[1]
                    correct[i] += pred.eq(target.view_as(pred)).sum().item()
            else:
                if args.rest_lyap:
                    train_loss += loss_lyap.mean()
                else:
                    train_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                
    train_loss /= len(train_loader.dataset)
    if args.individual_heads:
        correct = correct / len(train_loader.dataset)
        train_loss_str = ', '.join('{:.4f}'.format(loss) for loss in train_loss)
        correct_str = ', '.join(str(corr) for corr in correct)
        print('Train: Average loss: {}, Accuracy: {}%'.format(
        train_loss_str, 100. * correct_str))
        training_accuracy = correct
    else:
        print('Adv Train: Average loss: {:.4f}, Adv Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
        training_accuracy = correct / len(train_loader.dataset)
    
    return train_loss, training_accuracy

def adv_test(args, model, device, test_loader, attack_method = 'PGD'):
    model.eval()
    if args.individual_heads:
        test_loss = [0.0] * args.num_layers
        correct = [0] * args.num_layers
    else:
        test_loss = 0
        correct = 0
    if attack_method == 'PGD':
        attack = PGD(model, args.epsilon, args.step_size, args.num_steps, normalize=False)
    elif args.attack_type =='APGD':
        attack = APGD(model, eps=args.epsilon, seed=args.seed,device=device)
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if args.attack_type =='APGD':
            data_adv = attack.perturb(data, target)
        else:
            data_adv = attack(data, target)
        
        if args.test_rest_lyap:
            control = Lyap_Control(model, args.epsilon, args.step_size, args.num_steps)
            data_adv = control(data_adv)
        torch.cuda.empty_cache()
        with torch.no_grad():
            if args.rest_lyap:
                output, loss_lyap = model(data_adv, ret_lyap = True)
            else:
                output = model(data_adv)
                    
            if args.individual_heads:
                for i, x_out in enumerate(output):
                    
                    test_loss[i] += F.cross_entropy(x_out, target, size_average=False).item()
                    pred = output[i].max(1, keepdim=True)[1]
                    correct[i] += pred.eq(target.view_as(pred)).sum().item()
            else:
                if args.rest_lyap:
                    test_loss += loss_lyap.mean()
                else:
                    test_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
            torch.cuda.empty_cache()
            
    test_loss /= len(test_loader.dataset)
    if args.individual_heads:
        correct = correct / len(test_loader.dataset)
        test_loss_str = ', '.join('{:.4f}'.format(loss) for loss in test_loss)
        correct_str = ', '.join(str(corr) for corr in correct)
        print('Adv Test: Average loss: {}, Adv Accuracy: {}%'.format(
        test_loss_str, 100. * correct_str))
        test_accuracy = correct
        
    else:
        print('Adv Test: Average loss: {:.4f}, Adv Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if args.test_rest_lyap:
            control = Lyap_Control(model, args.epsilon, args.step_size, args.num_steps)
            data = control(data)
        with torch.no_grad():
            if args.rest_lyap:
                output, loss_lyap = model(data, ret_lyap = True)
            else:
                output = model(data)
                
            if args.individual_heads:
                for i, x_out in enumerate(output):
                    test_loss[i] += F.cross_entropy(x_out, target, size_average=False).item()
                    pred = output[i].max(1, keepdim=True)[1]
                    correct[i] += pred.eq(target.view_as(pred)).sum().item()
            else:
                if args.rest_lyap:
                        test_loss += loss_lyap.mean()
                else:
                    test_loss += F.cross_entropy(output, target, size_average=False).item()
                    
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        torch.cuda.empty_cache()
           
    test_loss /= len(test_loader.dataset)
    if args.individual_heads:
        correct = correct / len(test_loader.dataset)
        test_loss_str = ', '.join('{:.4f}'.format(loss) for loss in test_loss)
        correct_str = ', '.join(str(corr) for corr in correct)
        print('Test: Average loss: {}, Accuracy: {}%'.format(
        test_loss_str, 100. * correct_str))
        test_accuracy = correct
    else:
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        test_accuracy = correct / len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy
    