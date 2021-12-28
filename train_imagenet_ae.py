import argparse
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from FCN import *
from general_torch_model import GeneralTorchModel
from loss_utils import MarginLoss


os.environ["CUDA_VISIBLE_DEVICES"]='0'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--config', default='config/train_imagenet.json', help='config file')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    with open(args.config) as config_file:
        state = json.load(config_file)

    print(state)

    transform = transforms.Compose(
		[transforms.Resize(256),
		 transforms.CenterCrop(224),
		 transforms.ToTensor()])
    trainset = datasets.ImageFolder(state['train_path'], transform=transform)
    train_loader = data.DataLoader(trainset, batch_size=state['batch_size'], shuffle=True, num_workers=8)

    testset = datasets.ImageFolder(state['data_path'], transform=transform)
    test_loader = data.DataLoader(testset, batch_size=state['test_bs'], shuffle=False, num_workers=8)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    nets = []
    for model_name in state['model_name']:
        print(model_name)
        if model_name == "VGG16":
            pretrained_model = models.vgg16_bn(pretrained=True)
            pretrained_model = torch.nn.DataParallel(pretrained_model)
        elif model_name == 'Resnet18':
            pretrained_model = models.resnet18(pretrained=True)
            pretrained_model = torch.nn.DataParallel(pretrained_model)
        elif model_name == 'Squeezenet':
            pretrained_model = models.squeezenet1_1(pretrained=True)
            pretrained_model = torch.nn.DataParallel(pretrained_model)
        elif model_name == 'Googlenet':
            pretrained_model = models.googlenet(pretrained=True)
            pretrained_model = torch.nn.DataParallel(pretrained_model)
        pretrained_model.cuda().eval()
        net = GeneralTorchModel(pretrained_model,im_mean=mean,im_std=std)
        nets.append(net)
    if state['test_target']:
        target_nets = []
        for model_name in state['target_model_name']:
            if model_name == 'VGG19':
                target_model = models.vgg19(pretrained=True)
                target_model = torch.nn.DataParallel(target_model)
            if model_name == "resnet50":
                target_model = models.resnet50(pretrained=True)
                target_model = torch.nn.DataParallel(target_model)
            target_model.cuda().eval()
            target_net = GeneralTorchModel(target_model,im_mean=mean,im_std=std)
            target_nets.append(target_net)
    
    # print(nets)
    model = nn.Sequential(Imagenet_Encoder(),Imagenet_Decoder())
    model = torch.nn.DataParallel(model)
    model.cuda()

    optimizer_G = torch.optim.SGD(model.parameters(), state['learning_rate_G'], momentum=state['momentum'],
                                  weight_decay=0, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=state['epochs'] // state['schedule'],
                                                  gamma=state['gamma'])

    hingeloss = MarginLoss(margin=state['margin'], target=state['target'])

    if state['target']:
        save_name = "imagenet_{}{}_tanh_target_{}_{}.pth".format("_".join(state['model_name']), state['save_suffix'],
                                                             state['target_class'],state['norm'])
    else:
        save_name = "imagenet_{}{}_tanh_untarget_{}.pth".format("_".join(state['model_name']), state['save_suffix'],state['norm'])


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
        print(success)
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
            torch.save(model.module.state_dict(), os.path.join("G_weight","best_success_"+ save_name))

        torch.save(model.module.state_dict(), os.path.join("G_weight", save_name))
        print("epoch {}, Current success: {}, Best success: {}".format(epoch, state['test_success'], best_success))
