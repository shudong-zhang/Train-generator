import torch
import os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import cifar_models as models
from mnist_model import MNIST_target_1,MNIST_target_2
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from FCN import *
import torchvision.models as models
from general_torch_model import GeneralTorchModel

os.environ['CUDA_VISIBLE_DEVICES']='2'

transform=transforms.Compose([transforms.ToTensor(),
                                  ])
# mnist
# testset = datasets.MNIST("/data/zsd/dataset/mnist", train=False, download=False, transform=transform)
# test_loader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# pretrained_model = MNIST_target_2()
# pretrained_model = torch.nn.DataParallel(pretrained_model)
# checkpoint = torch.load('mnist_checkpoints/t2.pth')
# pretrained_model.load_state_dict(checkpoint['net'])
# pretrained_model.cuda()
# pretrained_model.eval()
# pretrained_model = GeneralTorchModel(pretrained_model,n_class=10,im_mean=None,im_std=None)

# model = nn.Sequential(MNIST_Encoder(),MNIST_Decoder())
# model.load_state_dict(torch.load('G_weight/best_success_MNIST_s1_s2_tanh_untarget.pth'))
# model = torch.nn.DataParallel(model)
# model.cuda()

# imagenet
transform = transforms.Compose(
		[transforms.Resize(256),
		 transforms.CenterCrop(224),
		 transforms.ToTensor()])
testset = datasets.ImageFolder('/data/zsd/dataset/imagenet/val_5000',transform=transform)
test_loader = data.DataLoader(testset,batch_size=25,shuffle=False)

pretrained_model = models.resnet50(pretrained=True)
# pretrained_model = models.vgg19_bn(pretrained=True)
pretrained_model = torch.nn.DataParallel(pretrained_model)
pretrained_model.cuda()
pretrained_model = GeneralTorchModel(pretrained_model,n_class=1000,im_mean=[0.4914, 0.4822, 0.4465],im_std=[0.2023, 0.1994, 0.2010])

model = nn.Sequential(Imagenet_Encoder(),Imagenet_Decoder())
# model.load_state_dict(torch.load('/data/zsd/attacks/black_box/Bayes_Trans_torch/vae/Imagenet_VGG16_Resnet18_Squeezenet_Googlenet_target_tanh_0.pytorch_best_success'))
model.load_state_dict(torch.load('/data/zsd/pytorch-train-generator/G_weight/imagenet_VGG16_Resnet18_Squeezenet_tanh_untarget.pth'))
model = torch.nn.DataParallel(model)
model.cuda().eval()


corr = 0
corr_n = 0
total = 0
for i,(x,y) in enumerate(test_loader):
  x = x.cuda()
  y = y.cuda()
  # target = 0
  
  x_adv = torch.clamp(x+0.03125*model(x),0,1)

  # label = pretrained_model.predict_label(x_adv)
  logits_n = pretrained_model(x)
  logits = pretrained_model(x_adv)

  # print(label.eq(y))
  corr += int((torch.argmax(logits, dim=1) == y).sum())
  corr_n += int((torch.argmax(logits_n, dim=1) == y).sum())
  # corr += label.eq(target).sum().item()
  
  total += y.size(0)

print(corr)
print(corr_n)
print(corr-corr_n)
print(total)
print(corr/total)