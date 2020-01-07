'''
WGAN-based adversarial auto-encoder

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
            nn.Linear(64, 9)
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
    def __init__(self, output_dim=1024, gpu=1):
        super(Sampler, self).__init__()
        self.output_dim = output_dim
        self.ff = nn.Sequential(
            nn.Linear(output_dim, output_dim, bias=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2)
        )
        self.mu_proj = nn.Linear(output_dim, output_dim)
        self.sig_proj = nn.Linear(output_dim, output_dim)
        self.gpu = gpu

    def forward(self, batch_size):
        '''Returns a sample from the Gaussian parametrized by the model
        '''
        u = torch.empty((batch_size, self.output_dim)).uniform_(-1,1).cuda(self.gpu)
        h = self.ff(u)
        mu = self.mu_proj(h)
        sig = self.sig_proj(h)
        eps = torch.normal(torch.zeros(sig.shape)).cuda(self.gpu)
        z_p = mu+sig*Variable(eps, requires_grad=False)
        z_p = torch.tanh(z_p)
        return z_p

class GAN(nn.Module):
    '''GAN Auto-encoder model
    '''
    def __init__(self, hidden_dim=1024, gpu=1):
        super(GAN, self).__init__()
        self.encoder = Encoder(output_dim=hidden_dim)
        self.decoder = Decoder(input_dim=hidden_dim)
        self.discr = Discriminator(input_dim=hidden_dim)
        self.sampler = Sampler(output_dim=hidden_dim, gpu=gpu)
        self.slp_start_i = 0
        self.slp_end_i = 128
        self.classifier = MLPClassifier(input_dim=self.slp_end_i-self.slp_start_i)

    def forward_enc_dec(self, x):
        '''Forward pass for the auto-encoder'''
        z = self.encoder(x)
        x_p = self.decoder(z, x)
        return x_p

    def forward_classifier(self, x):
        '''Forward pass for the classification task'''
        z = self.encoder(x)
        z = z.squeeze() # shape: (batch_size, 512)
        z = z[:, self.slp_start_i:self.slp_end_i]
        out = self.classifier(z)
        return out

    def forward_gen(self, batch_size):
        z_p = self.sampler(batch_size)
        return z_p

    def forward_enc(self, x):
        z = self.encoder(x).squeeze()
        return z

    def forward_discr(self, z):
        '''
        Args:
            z: shape (batch_size, hidden_dim)
        '''
        out = self.discr(z)
        return out

    def forward(self, x):
        return x

class DiscrLoss(nn.Module):
    '''Discriminative Loss

    0 for real, 1 for fake
    '''
    def __init__(self, lam=10.0, gpu=1):
        super(DiscrLoss, self).__init__()
        self.lam = lam
        self.gpu = gpu

    def forward(self, model, z, z_p):
        '''
        Args:
            z: real data, shape (batch_size, hidden_dim)
            z_p: fake data, shape (batch_size, hidden_dim)
        '''
        N = z_p.shape[0]
        eps = torch.rand(N, 1)
        eps = eps.expand(z.size()) # shape (batch_size, hidden_dim)
        eps = eps.cuda(self.gpu)
        z_hat = eps * z + ((1 - eps) * z_p)
        z_hat = z_hat.cuda(self.gpu)
        z_hat = Variable(z_hat, requires_grad=True)
        d_z = model.forward_discr(z) # shape (batch_size, 2)
        d_z_hat = model.forward_discr(z_hat) # shape (batch_size, 2)
        d_z = F.softmax(d_z, 1)[:, 1]
        d_z_hat = F.softmax(d_z_hat, 1)[:, 1]
        gradients = grad(outputs=d_z_hat, inputs=z_hat, \
            grad_outputs=torch.ones(d_z_hat.size()).cuda(self.gpu), \
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return d_z_hat.mean() - d_z.mean() + self.lam*gradient_penalty
