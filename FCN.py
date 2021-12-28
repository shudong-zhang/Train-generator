# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Imagenet_Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        return torch.tanh(x)

class Imagenet_Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.deconv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.deconv1_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.deconv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.deconv2_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.deconv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.deconv3_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.deconv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.deconv4_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        self.fcn = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        n = len(x)
        x = x.view(n, 3, 14, 14)

        y = self.deconv1_1(x)
        y = self.deconv1_2(y)
        y = self.deconv2_1(y)
        y = self.deconv2_2(y)
        y = self.deconv3_1(y)
        y = self.deconv3_2(y)
        y = self.deconv4_1(y)
        y = self.deconv4_2(y)
        y = self.fcn(y)

        return torch.tanh(y)

class CIFAR10_Encoder(nn.Module):


    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        return torch.tanh(x)

class CIFAR10_Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.deconv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.deconv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.deconv2_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.deconv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.deconv2_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        self.deconv2_4 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)

    def forward(self, x):
        n = len(x)
        x = x.view(n, 3, 8, 8)

        y = self.deconv1_1(x)
        y = self.deconv1_2(y)
        y = self.deconv2_1(y)
        y = self.deconv2_2(y)
        y = self.deconv2_3(y)
        y = self.deconv2_4(y)

        return torch.tanh(y)

class MNIST_Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        return torch.tanh(x)

class MNIST_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.deconv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.deconv2_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.deconv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.deconv2_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        self.deconv2_4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        n = len(x)
        x = x.view(n, 1, 7, 7)

        y = self.deconv1_1(x)
        y = self.deconv1_2(y)
        y = self.deconv2_1(y)
        y = self.deconv2_2(y)
        y = self.deconv2_3(y)
        y = self.deconv2_4(y)

        return torch.tanh(y)


if __name__ == "__main__":
    model1 = CIFAR10_Encoder()
    model2 = CIFAR10_Decoder()
    model3 = MNIST_Encoder()
    model4 = MNIST_Decoder()
    x = torch.randn(size=(1,3,32,32))
    x_mni = torch.randn(size=(1,1,28,28))
    # print(x.shape)
    y = model1(x)
    x_mni_latent = model3(x_mni)
    print(x_mni_latent.shape)
    # print(y.shape)
    # x_ = model2(y)
    # print(x_.shape)

