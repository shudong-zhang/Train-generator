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
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

# os.environ["CUDA_VISIBLE_DEVICES"]='0'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0,1')
parser.add_argument('--config', default='config/train_imagenet.json', help='config file')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
with open(args.config) as config_file:
    state = json.load(config_file)

print(state)

def main():
    
    nprocs = torch.cuda.device_count()
    assert state['batch_size'] % nprocs == 0
    assert state['test_bs'] % nprocs == 0
    mp.spawn(main_worker, nprocs=nprocs, args=(nprocs, state))

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def main_worker(local_rank, nprocs, state):
    print(local_rank)
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:34203', world_size=nprocs, rank=local_rank)
    state['batch_size'] = int(state['batch_size'] / nprocs)
    state['test_bs'] = int(state['test_bs'] / nprocs)
    transform = transforms.Compose(
		[transforms.Resize(256),
		 transforms.CenterCrop(224),
		 transforms.ToTensor()])
    trainset = datasets.ImageFolder(state['train_path'], transform=transform)
    testset = datasets.ImageFolder(state['data_path'], transform=transform)
    trainsampler = DistributedSampler(trainset,shuffle=True)
    testsampler = DistributedSampler(testset,shuffle=False)
    train_loader = data.DataLoader(trainset, batch_size=state['batch_size'], num_workers=0,sampler=trainsampler)
    test_loader = data.DataLoader(testset, batch_size=state['test_bs'], num_workers=0,sampler=testsampler)

    torch.cuda.set_device(local_rank)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    nets = []
    for model_name in state['model_name']:
        print(model_name)
        if model_name == "VGG16":
            pretrained_model = models.vgg16_bn(pretrained=True)
        elif model_name == 'Resnet18':
            pretrained_model = models.resnet18(pretrained=True)
        elif model_name == 'Squeezenet':
            pretrained_model = models.squeezenet1_1(pretrained=True)
        elif model_name == 'Googlenet':
            pretrained_model = models.googlenet(pretrained=True)
        pretrained_model.cuda(local_rank).eval()
        pretrained_model = nn.SyncBatchNorm.convert_sync_batchnorm(pretrained_model).to(local_rank)
        pretrained_model = DistributedDataParallel(pretrained_model, device_ids=[local_rank])
        net = GeneralTorchModel(pretrained_model,im_mean=mean,im_std=std)
        nets.append(net)
    if state['test_target']:
        target_nets = []
        for model_name in state['target_model_name']:
            if model_name == 'VGG19':
                target_model = models.vgg19_bn(pretrained=True)
            if model_name == "resnet50":
                target_model = models.resnet50(pretrained=True)
            target_model.cuda(local_rank).eval()
            target_model = nn.SyncBatchNorm.convert_sync_batchnorm(target_model).to(local_rank)
            target_model = DistributedDataParallel(target_model, device_ids=[local_rank])
            target_net = GeneralTorchModel(target_model,im_mean=mean,im_std=std)
            target_nets.append(target_net)
    
    # print(nets)
    model = nn.Sequential(Imagenet_Encoder(),Imagenet_Decoder())
    # model = torch.nn.DataParallel(model)
    model.cuda(local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer_G = torch.optim.SGD(model.parameters(), state['learning_rate_G'], momentum=state['momentum'],
                                  weight_decay=0, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=state['epochs'] // state['schedule'],
                                                  gamma=state['gamma'])

    hingeloss = MarginLoss(margin=state['margin'], target=state['target'])

    if state['target']:
        save_name = "imagenet_{}{}_tanh_target_{}.pth".format("_".join(state['model_name']), state['save_suffix'],
                                                             state['target_class'])
    else:
        save_name = "imagenet_{}{}_tanh_untarget.pth".format("_".join(state['model_name']), state['save_suffix'])


    def train():
        model.train()

        for batch_idx, (data, label) in enumerate(train_loader):
            nat = data.cuda(local_rank, non_blocking=True)
            if state['target']:
                label = state['target_class']
            else:
                label = label.cuda(local_rank, non_blocking=True)

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
                torch.distributed.barrier()
                reduced_loss = reduce_mean(loss_g, nprocs)
                losses_g.append("%4.2f" % reduced_loss.item())
                loss_g.backward()
            optimizer_G.step()
            if local_rank == 0:
                if (batch_idx + 1) % state['log_interval'] == 0:
                    print("batch {}, losses_g {}".format(batch_idx + 1, dict(zip(state['model_name'], losses_g))))
    def normalize(x):
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return x / (t.view(-1,1,1,1) + 1e-12)

    def test():
        model.eval()
        loss_avg = [0.0 for i in range(len(nets))]
        success = [0 for i in range(len(nets))]
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                nat = data.cuda(local_rank, non_blocking=True)
                if state['target']:
                    label = state['target_class']
                else:
                    label = label.cuda(local_rank, non_blocking=True)
                noise = model(nat)
                if state['norm'] == 'linf':
                    adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
                elif state['norm'] == 'l2':
                    adv = torch.clamp(nat + normalize(noise) * state['epsilon'], 0. ,1.)

                for j in range(len(nets)):
                    logits = nets[j](adv)
                    loss = hingeloss(logits, label)
                    torch.distributed.barrier()
                    reduced_loss = reduce_mean(loss, nprocs)
                    loss_avg[j] += reduced_loss.item()
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
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                nat = data.cuda(local_rank, non_blocking=True)
                if state['target']:
                    label = state['target_class']
                else:
                    label = label.cuda(local_rank, non_blocking=True)
                noise = model(nat)
                if state['norm'] == 'linf':
                    adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
                elif state['norm'] == 'l2':
                    adv = torch.clamp(nat + normalize(noise) * state['epsilon'], 0. ,1.)
                for j in range(len(target_nets)):
                    logits = target_nets[j](adv)
                    loss = hingeloss(logits, label)
                    torch.distributed.barrier()
                    reduced_loss = reduce_mean(loss, nprocs)
                    loss_avg[j] += reduced_loss.item()
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
        trainsampler.set_epoch(epoch)
        testsampler.set_epoch(epoch)
        state['epoch'] = epoch
        train()
        scheduler_G.step()
        torch.cuda.empty_cache()
        test_target()
        if local_rank == 0:
            print(state)
            if best_success<state['test_success']:
                best_success = state['test_success']
                torch.save(model.module.state_dict(), os.path.join("G_weight","best_success_"+ save_name))

            torch.save(model.module.state_dict(), os.path.join("G_weight", save_name))
            print("epoch {}, Current success: {}, Best success: {}".format(epoch, state['test_success'], best_success))

if __name__ == "__main__":
    main()