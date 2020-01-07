'''
Based on https://github.com/gabrielhuang/reptile-pytorch

Peter Wu
peterw1@andrew.cmu.edu
'''

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

from alexnet import alexnet


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SimpleCNN(nn.Module):
    def __init__(self, output_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.relu1 = nn.ReLU(inplace=True) # nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True) # nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(64*4, output_dim)
        )

    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, 3, 360, 240)
        '''
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (2, 2))
        x = x.view(len(x), 256)
        x = self.fc(x)
        return x


def mk_simple_cnn(output_dim):
    net = SimpleCNN(output_dim)
    net.apply(init_weights)
    return net


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, vocab_size, emb_mat, bidirectional=True):
        '''
        Args:
            vocab_size: note might not equal emb_vocab_size
                always >= emb_vocab_size though
                includes pad token and unknown word token
            emb_mat: numpy array with shape (emb_vocab_size, emb_dim)
        '''
        super(SimpleLSTM, self).__init__()
        emb_dim = emb_mat.shape[1]
        self.emb_layer = nn.Embedding(vocab_size, emb_dim)
        num_new_vocab = vocab_size - emb_mat.shape[0]
        extra_embs = np.random.normal(0.0, 1.0, size=(num_new_vocab, emb_dim))
        new_emb_mat = np.concatenate([emb_mat, extra_embs], 0)
        self.emb_layer.weight.data.copy_(torch.from_numpy(new_emb_mat))
        self.lstm = nn.LSTM(emb_dim, hidden_dim, 1, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, seq_len)
        '''
        x = self.emb_layer(x) # shape (batch_size, seq_len, emb_dim)
        x, (c,h) = self.lstm(x, None) # h shape (2, batch_size, hidden_dim)
        hidden_left, hidden_right = h[0,:,:], h[1,:,:]
        hidden = torch.cat((hidden_left, hidden_right),1)
        x = self.fc(hidden)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class ReptileModel(nn.Module):

    def __init__(self, device):
        nn.Module.__init__(self)
        self.device = device

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda(self.device)
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class OmniglotModel(ReptileModel):
    def __init__(self, fc_dim, num_classes, vocab_size, emb_mat, device):
        ReptileModel.__init__(self, device)

        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.emb_mat = emb_mat
        
        self.clf = alexnet(pretrained=False, progress=True, num_classes=int(fc_dim/2))

        self.net_text = SimpleLSTM(64, fc_dim, vocab_size, emb_mat)
        self.net_out = MLP(fc_dim, int(fc_dim/2))
        
        self.net_shared = MLP(int(fc_dim/2), num_classes)
        
    def forward_cca(self, x1, x2):
        out1 = self.clf(x1)

        x = self.net_text(x2)
        out2 = self.net_out(x)

        return out1, out2

    def forward(self, x):
        x = self.net_text(x)
        out2 = self.net_out(x)
        out = self.net_shared(out2)
        return out

    def predict(self, prob):
        __, argmax = prob.max(1)
        return argmax

    def clone(self):
        clone = OmniglotModel(self.fc_dim, self.num_classes, self.vocab_size, self.emb_mat, self.device)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda(self.device)
        return clone
