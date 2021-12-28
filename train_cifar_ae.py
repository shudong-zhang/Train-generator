import argparse
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import *
# from mnist_model import *
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from FCN import *
from general_torch_model import GeneralTorchModel
from loss_utils import MarginLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('--config', default='config/train_cifar.json', help='config file')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    with open(args.config) as config_file:
        state = json.load(config_file)

    print(state)

    transform=transforms.Compose([       
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                                  ])
    trainset = datasets.CIFAR10(state['data_path'], train=True, download=True, transform=transform)
    train_loader = data.DataLoader(trainset, batch_size=state['batch_size'], shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(state['data_path'], train=False, download=False, transform=transforms.ToTensor())
    test_loader = data.DataLoader(testset, batch_size=state['test_bs'], shuffle=False, num_workers=4)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    nets = []
    for model_name in state['model_name']:
        if model_name == 'vgg16':
            pretrained_model = VGG('VGG16')
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            checkpoint = torch.load(os.path.join(state['model_path'], model_name + '_ckpt.pth'))
            pretrained_model.load_state_dict(checkpoint['net'])
            pretrained_model.cuda()
            pretrained_model.eval()
        elif model_name == "senet":
            pretrained_model = SENet18()
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            checkpoint = torch.load(os.path.join(state['model_path'],model_name+'_ckpt.pth'))
            pretrained_model.load_state_dict(checkpoint['net'])
            pretrained_model.cuda()
            pretrained_model.eval()
        elif model_name == 'densenet':
            pretrained_model = DenseNet121()
            pretrained_model = torch.nn.DataParallel(pretrained_model)

            checkpoint = torch.load(os.path.join(state['model_path'], model_name+'_ckpt.pth'))
            pretrained_model.load_state_dict(checkpoint['net'])
            pretrained_model.cuda()
            pretrained_model.eval()
        net = GeneralTorchModel(pretrained_model,im_mean=mean,im_std=std)
        nets.append(net)
    if state['test_target']:
        target_nets = []
        for model_name in state['target_model_name']:
            if model_name == 'vgg19':
                target_model = VGG('VGG19')
                target_model = torch.nn.DataParallel(target_model)
                checkpoint = torch.load(os.path.join(state['model_path'],model_name+'_ckpt.pth'))
                target_model.load_state_dict(checkpoint['net'])
                target_model.cuda()
                target_model.eval()
            elif model_name == "resnet18":
                # target_model = resnet(depth=20,num_classes=10)
                target_model = ResNet18()
                target_model = torch.nn.DataParallel(target_model)
                # checkpoint = torch.load('cifar_checkpoints/resnet-20.tar')
                checkpoint = torch.load(os.path.join(state['model_path'], model_name+'_ckpt.pth'))
                # checkpoint = torch.load('cifar_checkpoints/resnet18_ckpt.pth')
                target_model.load_state_dict(checkpoint['net'])
                target_model.cuda()
                target_model.eval()

            target_net = GeneralTorchModel(target_model,im_mean=mean,im_std=std)
            target_nets.append(target_net)
    
    # print(nets)
    model = nn.Sequential(CIFAR10_Encoder(),CIFAR10_Decoder())
    model = torch.nn.DataParallel(model)
    model.cuda()

    optimizer_G = torch.optim.SGD(model.parameters(), state['learning_rate_G'], momentum=state['momentum'],
                                  weight_decay=0, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=state['epochs'] // state['schedule'],
                                                  gamma=state['gamma'])
    hingeloss = MarginLoss(margin=state['margin'], target=state['target'])

    if state['target']:
        save_name = "CIFAR_{}{}_tanh_target_{}_{}.pth".format("_".join(state['model_name']), state['save_suffix'],
                                                             state['target_class'],state['norm'])
    else:
        save_name = "CIFAR_{}{}_tanh_untarget_{}.pth".format("_".join(state['model_name']), state['save_suffix'],state['norm'])


    def train():
        model.train()

        for batch_idx, (data, label) in enumerate(train_loader):
            nat = data.cuda()
            if state['target']:
                label = state['target_class']
            else:
                label = label.cuda()

            losses_g = []
            optimizer_G.zero_grad()
            for net in nets:
                noise = model(nat)
                if state['norm'] == 'linf':
                    adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
                elif state['norm'] == 'l2':
                    adv = torch.clamp(nat + normalize(noise) * state['epsilon'], 0. ,1.)
                logits_adv = net(adv)
                loss_g = hingeloss(logits_adv, label)
                losses_g.append("%4.2f" % loss_g.item())
                loss_g.backward()
            optimizer_G.step()

            if (batch_idx + 1) % state['log_interval'] == 0:
                print("batch {}, losses_g {}".format(batch_idx + 1, dict(zip(state['model_name'], losses_g))))
    def normalize(x):
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return x / (t.view(-1,1,1,1) + 1e-12)

    def test():
        model.eval()
        loss_avg = [0.0 for i in range(len(nets))]
        success = [0 for i in range(len(nets))]

        for batch_idx, (data, label) in enumerate(test_loader):
            nat = data.cuda()
            if state['target']:
                label = state['target_class']
            else:
                label = label.cuda()
            noise = model(nat)
            if state['norm'] == 'linf':
                adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
            elif state['norm'] == 'l2':
                adv = torch.clamp(nat + normalize(noise) * state['epsilon'], 0. ,1.)

            for j in range(len(nets)):
                logits = nets[j](adv)
                loss = hingeloss(logits, label)
                loss_avg[j] += loss.item()
                if state['target']:
                    success[j] += int((torch.argmax(logits, dim=1) == label).sum())
                else:
                    success[j] += int((torch.argmax(logits, dim=1) != label).sum())

        state['test_loss'] = [loss_avg[i] / len(test_loader) for i in range(len(loss_avg))]
        state['test_successes'] = [success[i] / len(test_loader.dataset) for i in range(len(success))]
        state['test_success'] = 0.0
        for i in range(len(state['test_successes'])):
            state['test_success'] += state['test_successes'][i] / len(state['test_successes'])

    def test_target():
        model.eval()
        
        loss_avg = [0.0 for i in range(len(target_nets))]
        success = [0 for i in range(len(target_nets))]
        
        for batch_idx, (data, label) in enumerate(test_loader):
            nat = data.cuda()
            if state['target']:
                label = state['target_class']
            else:
                label = label.cuda()
            noise = model(nat)
            if state['norm'] == 'linf':
                adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
            elif state['norm'] == 'l2':
                adv = torch.clamp(nat + normalize(noise) * state['epsilon'], 0. ,1.)

            for j in range(len(target_nets)):
                logits = target_nets[j](adv)
                loss = hingeloss(logits, label)
                loss_avg[j] += loss.item()
                if state['target']:
                    success[j] += int((torch.argmax(logits, dim=1) == label).sum())
                else:
                    success[j] += int((torch.argmax(logits, dim=1) != label).sum())

        state['test_loss'] = [loss_avg[i] / len(test_loader) for i in range(len(loss_avg))]
        state['test_successes'] = [success[i] / len(test_loader.dataset) for i in range(len(success))]
        state['test_success'] = 0.0
        for i in range(len(state['test_successes'])):
            state['test_success'] += state['test_successes'][i] / len(state['test_successes'])


    best_success = 0.0
    for epoch in range(state['epochs']):
        state['epoch'] = epoch
        train()
        scheduler_G.step()
        torch.cuda.empty_cache()
        test_target()
        print(state)
        if best_success<state['test_success']:
            best_success = state['test_success']
            torch.save(model.module.state_dict(), os.path.join("ae_checkpoint","best_success_"+ save_name))

        torch.save(model.module.state_dict(), os.path.join("ae_checkpoint", save_name))
        print("epoch {}, Current success: {}, Best success: {}".format(epoch, state['test_success'], best_success))


