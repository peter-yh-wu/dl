'''Conditional GAN

Implementation based on https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN/blob/master/pytorch_MNIST_cDCGAN.py

Peter Wu
peterw1@andrew.cmu.edu
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self, hidden_dim=100, cond_emb_dim=10, d=128):
        super(Generator, self).__init__()

        self.deconv1_1 = nn.ConvTranspose2d(hidden_dim, d*2, (4,2), (1,1), (0,0)) # 1->4, 1->2
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(cond_emb_dim, d*2, (4,2), (1,1), (0,0)) # 1->4, 1->2
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1) # 4->8, 2->4
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1) # 8->16, 4->8
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1) # 16->32, 8->16
        self.deconv4_bn = nn.BatchNorm2d(1)
        self.deconv5 = nn.ConvTranspose2d(1, 1, 4, 2, 1) # 32->64, 16->32
        self.deconv5_bn = nn.BatchNorm2d(1)
        self.deconv6 = nn.ConvTranspose2d(1, 1, 4, 2, 1) # 64->128, 32->64

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, inp, label, ref):
        '''
        Args:
            inp: (batch_size, hidden_dim, 1, 1)
            label: (batch_size, cond_emb_dim, 1, 1)
            ref: same shape as output
        '''
        x = self.deconv1_1(inp)
        x = F.relu(self.deconv1_1_bn(x))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = (self.deconv6(x)+ref).tanh()
        return x

class Discriminator(nn.Module):
    def __init__(self, cond_emb_dim=10, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(cond_emb_dim, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, 1, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, inp, label):
        '''
        Args:
            inp: 64, 1, 512, 80
            label: 64, cond_emb_dim, 512, 80
        '''
        x = F.leaky_relu(self.conv1_1(inp), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = self.conv3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.sigmoid()
        return x.squeeze(3).squeeze(2)