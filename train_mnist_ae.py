import argparse
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import cifar_models as models
from mnist_model import *
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from FCN import *
# from general_torch_model import GeneralTorchModel
from loss_utils import MarginLoss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--device', type=int, nargs="+", default=[0,1])
    parser.add_argument('--config', default='config/train_mnist.json', help='config file')

    args = parser.parse_args()
    with open(args.config) as config_file:
        state = json.load(config_file)

    print(state)

    transform=transforms.Compose([transforms.ToTensor(),
                                  ])
    trainset = datasets.MNIST(state['data_path'], train=True, download=True, transform=transform)
    train_loader = data.DataLoader(trainset, batch_size=state['batch_size'], shuffle=True, num_workers=8)

    testset = datasets.MNIST(state['data_path'], train=False, download=False, transform=transform)
    test_loader = data.DataLoader(testset, batch_size=state['test_bs'], shuffle=False, num_workers=8)

    nets = []
    for model_name in state['model_name']:
        if model_name == 's1':
            pretrained_model = MNIST_s1()
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            checkpoint = torch.load(os.path.join(state['model_path'],model_name+'.pth'))
            pretrained_model.load_state_dict(checkpoint['net'])
            pretrained_model.cuda()
            pretrained_model.eval()
        if model_name == "s2":
            pretrained_model = MNIST_s2()
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            checkpoint = torch.load(os.path.join(state['model_path'],model_name+'.pth'))
            pretrained_model.load_state_dict(checkpoint['net'])
            pretrained_model.cuda()
            pretrained_model.eval()

        # net = GeneralTorchModel(pretrained_model,im_mean=mean,im_std=std)
        nets.append(pretrained_model)
    if state['test_target']:
        target_nets = []
        for model_name in state['target_model_name']:
            if model_name == 't1':
                target_model = MNIST_target_1()
                target_model = torch.nn.DataParallel(target_model)
                checkpoint = torch.load(os.path.join(state['model_path'],model_name+'.pth'))
                target_model.load_state_dict(checkpoint['net'])
                target_model.cuda()
                target_model.eval()
            if model_name == "t2":
                target_model = MNIST_target_2()
                target_model = torch.nn.DataParallel(target_model)
                checkpoint = torch.load(os.path.join(state['model_path'],model_name+'.pth'))
                target_model.load_state_dict(checkpoint['net'])
                target_model.cuda()
                target_model.eval()

            # net = GeneralTorchModel(pretrained_model,im_mean=mean,im_std=std)
            target_nets.append(target_model)
    
    # print(nets)
    model = nn.Sequential(MNIST_Encoder(),MNIST_Decoder())
    model = torch.nn.DataParallel(model)
    model.cuda()

    optimizer_G = torch.optim.SGD(model.parameters(), state['learning_rate_G'], momentum=state['momentum'],
                                  weight_decay=0, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=state['epochs'] // state['schedule'],
                                                  gamma=state['gamma'])

    hingeloss = MarginLoss(margin=state['margin'], target=state['target'])

    if state['target']:
        save_name = "MNIST_{}{}_tanh_target_{}.pth".format("_".join(state['model_name']), state['save_suffix'],
                                                             state['target_class'])
    else:
        save_name = "MNIST_{}{}_tanh_untarget.pth".format("_".join(state['model_name']), state['save_suffix'])


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
                adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
                logits_adv = net(adv)
                loss_g = hingeloss(logits_adv, label)
                losses_g.append("%4.2f" % loss_g.item())
                loss_g.backward()
            optimizer_G.step()

            if (batch_idx + 1) % state['log_interval'] == 0:
                print("batch {}, losses_g {}".format(batch_idx + 1, dict(zip(state['model_name'], losses_g))))


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
            adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)

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
            adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)

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
        scheduler_G.step()
        state['epoch'] = epoch
        train()
        torch.cuda.empty_cache()
        test_target()
        print(state)
        if best_success<state['test_success']:
            best_success = state['test_success']
            torch.save(model.module.state_dict(), os.path.join("G_weight","best_success_"+ save_name))

        torch.save(model.module.state_dict(), os.path.join("G_weight", save_name))
        print("epoch {}, Current success: {}, Best success: {}".format(epoch, state['test_success'], best_success))


