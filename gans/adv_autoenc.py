'''
Custom adversarial auto-encoder model

Peter Wu
peterw1@andrew.cmu.edu
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable, grad


class Encoder(nn.Module):
    '''CNN Encoder'''
    def __init__(self, output_dim=1024):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 4, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, output_dim, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    '''Decoder composed of deconvolutional layers'''
    def __init__(self, input_dim=1024):
        super(Decoder, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, 16, 2, 0, bias=False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 4, 2, 4, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, z, x_orig):
        '''
        Args:
            z: shape (batch_size, 512, 1, 1)
        '''
        z = self.generator(z)
        z = F.adaptive_max_pool2d(z, (x_orig.size()[2], x_orig.size()[3]))
        z = torch.tanh(z)
        return z

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024):
        super(MLPClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, z):
        out = self.mlp(z)
        return out

class Discriminator(nn.Module):
    '''MLP Discriminator

    Distinguishes real and fake latent representations; 0 for real, 1 for fake
    '''
    def __init__(self, input_dim=1024):
        super(Discriminator, self).__init__()
        self.discr = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, z):
        out = self.discr(z)
        return out

class Sampler(nn.Module):
    '''Outputs a generated version of the latent representation

    Assumes that the latent representation follows a Gaussian distribution
        with mean mu and standard deviation sig, where mu and sig are
        learned by the model
    '''
    def __init__(self, output_dim=1024):
        super(Sampler, self).__init__()
        self.output_dim = output_dim
        self.ff = nn.Sequential(
            nn.Linear(output_dim, output_dim, bias=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2)
        )
        self.mu_proj = nn.Linear(output_dim, output_dim)
        self.sig_proj = nn.Linear(output_dim, output_dim)

    def forward(self, batch_size):
        '''Returns a sample from the Gaussian parametrized by the model
        '''
        u = torch.empty((batch_size, self.output_dim)).uniform_(-1,1).cuda()
        h = self.ff(u)
        mu = self.mu_proj(h)
        sig = self.sig_proj(h)
        eps = torch.normal(torch.zeros(sig.shape)).cuda()
        z_p = mu+sig*Variable(eps, requires_grad=False)
        z_p = torch.tanh(z_p)
        return z_p

class GAN(nn.Module):
    '''GAN Auto-encoder model
    '''
    def __init__(self, hidden_dim=1024):
        super(GAN, self).__init__()
        self.encoder = Encoder(output_dim=hidden_dim)
        self.decoder = Decoder(input_dim=hidden_dim)
        self.classifier = MLPClassifier(input_dim=hidden_dim)
        self.discr = Discriminator(input_dim=hidden_dim)
        self.sampler = Sampler(output_dim=hidden_dim)

    def forward_enc_dec(self, x):
        '''Forward pass for the auto-encoder'''
        z = self.encoder(x)
        x_p = self.decoder(z, x)
        return x_p

    def forward_classifier(self, x):
        '''Forward pass for the classification task'''
        z = self.encoder(x)
        z = z.squeeze() # shape: (batch_size, 512)
        out = self.classifier(z)
        return out

    def forward_gen(self, batch_size):
        '''Forward pass for the Generative Loss

        i.e. the Sampler attempts to fool the Discriminator
        '''
        z_p = self.sampler(batch_size)
        out_p = self.discr(z_p)
        return out_p

    def forward_discr(self, x):
        '''Forward pass for the Discriminative Loss

        i.e. the Discriminator attempts to identify which samples are real and fake
        '''
        z = self.encoder(x).squeeze()
        out = self.discr(z)
        out_p = self.forward_gen(z.shape[0])
        return out, out_p

    def forward(self, x):
        return x

class DiscrLoss(nn.Module):
    '''Discriminative Loss

    0 for real, 1 for fake
    '''
    def __init__(self, frac_flip=0.1):
        '''
        Args:
            frac_flip: the fraction of samples that will have their labels
                flipped; labels are flipped to make inputs noisy and thus
                prevent the Discriminator from becoming too powerful
        '''
        super(DiscrLoss, self).__init__()
        self.frac_flip = frac_flip

    def forward(self, out, out_p):
        '''
        Args:
            out: shape (batch_size, 2)
            out_p: shape (batch_size, 2)
        '''
        N = out_p.shape[0]
        out = F.softmax(out, 1)
        out_p = F.softmax(out_p, 1)
        shuffled_indices = torch.randperm(N)
        num_diff = int(self.frac_flip*N)
        diff_indices = shuffled_indices[:num_diff]
        same_indices = shuffled_indices[num_diff:]
        loss1_same = -torch.sum(torch.log(out[same_indices, 0]))
        loss1_diff = -torch.sum(torch.log(out[diff_indices, 1]))
        loss2_same = -torch.sum(torch.log(out_p[same_indices, 1]))
        loss2_diff = -torch.sum(torch.log(out_p[diff_indices, 0]))
        loss = (loss1_same + loss1_diff + loss2_same + loss2_diff)/2/N
        return loss

class GenLoss(nn.Module):
    '''Generative Loss
    '''
    def __init__(self, frac_flip=0.0):
        '''
        Args:
            frac_flip: the fraction of samples that will have their labels
                flipped; labels are flipped to make inputs noisy and thus
                prevent the Generator from becoming too powerful
        '''
        super(GenLoss, self).__init__()
        self.frac_flip = frac_flip

    def forward(self, out_p):
        '''
        Args:
            out_p: shape (batch_size, 2)
        '''
        N = out_p.shape[0]
        out_p = F.softmax(out_p, 1)
        shuffled_indices = torch.randperm(N)
        num_diff = int(self.frac_flip*N)
        diff_indices = shuffled_indices[:num_diff]
        same_indices = shuffled_indices[num_diff:]
        loss_same = -torch.sum(torch.log(out_p[same_indices, 0]))
        loss_diff = -torch.sum(torch.log(out_p[diff_indices, 1]))
        loss = (loss_same+loss_diff)/N
        return loss