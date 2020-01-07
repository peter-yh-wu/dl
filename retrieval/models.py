'''
Peter Wu
peterw1@andrew.cmu.edu
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F


def cosine_sim(im, s):
    '''Cosine similarity between all the image and sentence pairs
    '''
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):
    '''
    Compute contrastive loss
    '''
    def __init__(self, margin=0, max_violation=False, cuda=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.cuda = cuda

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda(self.cuda)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class MelLSTMClf(nn.Module):
    def __init__(self, input_size, model_hidden_size, model_embedding_size, model_num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size)
    
    def forward(self, utterances, hidden_init=None):
        _, (hidden, cell) = self.lstm(utterances, hidden_init)
        embeds_raw = self.linear(hidden[-1])
        return embeds_raw


class SimpleDeviseModel(nn.Module):
    def __init__(self, audio_dim, enc2, model_hidden_size, model_embedding_size, model_num_layers, text_dim=768):
        super(SimpleDeviseModel, self).__init__()
        self.text_enc = nn.Linear(text_dim, model_embedding_size)
        if enc2 == 'lstm':
            self.mel_enc = MelLSTMClf(audio_dim, model_hidden_size, model_embedding_size, model_num_layers)
        else:
            self.mel_enc = CNNClf(model_embedding_size)

    def forward(self, x_text, x_spkr):
        out_text = self.text_enc(x_text)
        out_mel = self.mel_enc(x_spkr)
        return out_text, out_mel


class CNNClf(nn.Module):
    def __init__(self, output_dim):
        super(CNNClf, self).__init__()
        
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv(x)
        out = F.adaptive_avg_pool2d(out, (2, 2))
        out = out.view(len(out), -1)
        out = self.classifier(out)
        return out


class CNNEncDec(nn.Module):
    def __init__(self):
        super(CNNEncDec, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(1)
        return x
