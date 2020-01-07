'''
Peter Wu
peterw1@andrew.cmu.edu
'''

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        '''model_type: funnel or block'''
        super(MLP, self).__init__()
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        model_type = args.model_type
        dropout = args.dropout
        self.use_softmax = args.softmax

        hidden_dims = [hidden_dim for _ in range(num_layers)]
        if model_type == 'funnel':
            log_input_dim = int(math.log(input_dim, 2))
            log_output_dim = int(math.log(output_dim, 2))
            delta = (log_input_dim-log_output_dim)/(num_layers+1)
            log_hidden_dims = [log_input_dim-delta*(i+1) for i in range(num_layers)]
            hidden_dims = [int(math.pow(2, l)) for l in log_hidden_dims]
        elif model_type == 'triangle':
            delta = (input_dim-output_dim)/(num_layers+1)
            hidden_dims = [int(input_dim-delta*(i+1)) for i in range(num_layers)]
        dims = [input_dim]+hidden_dims
        self.fc_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.Dropout(dropout), nn.ReLU()) \
            for i in range(num_layers)])
        self.output = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for i, l in enumerate(self.fc_layers):
            x = self.fc_layers[i](x)
        x = self.output(x)
        if self.use_softmax:
            x = F.softmax(x, 1)
        return x


class Simple_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(Simple_LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.bidirectional = args.bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.seq_model = nn.LSTM(self.input_dim, self.hidden_dim, 
            self.num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.output = nn.Linear(self.hidden_dim*self.num_directions, self.output_dim)

    def init_hidden(self, batch_size):
        return(Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim)),
            Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim)))

    def forward(self, x):
        '''x has shape (batch_size, seq_len, num_feats)'''
        o, (h, c) = self.seq_model(x, self.init_hidden(x.shape[0]))
        h = h.view(self.num_layers, self.num_directions, x.shape[0], self.hidden_dim)
        if self.bidirectional:
            h = torch.cat((h[-1, 0, :, :], h[-1, 1, :, :]), 1)
        else:
            h = h[-1, 0, :, :]
        x = self.output(h)
        return x
