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

class MLP_TriSlab(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        '''model_type: funnel or block'''
        super(MLP_TriSlab, self).__init__()
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        model_type = args.model_type
        dropout = args.dropout
        self.emb_dim = args.twin_dim

        hidden_dims = [hidden_dim for _ in range(num_layers)]
        if model_type == 'funnel':
            log_input_dim = int(math.log(input_dim, 2))
            log_output_dim = int(math.log(self.emb_dim, 2))
            delta = (log_input_dim-log_output_dim)/(num_layers+1)
            log_hidden_dims = [log_input_dim-delta*(i+1) for i in range(num_layers)]
            hidden_dims = [int(math.pow(2, l)) for l in log_hidden_dims]
        dims = [input_dim]+hidden_dims
        self.fc_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.Dropout(dropout), nn.ReLU()) \
            for i in range(num_layers)])
        self.emb_output = nn.Sequential(nn.Linear(dims[-1], self.emb_dim), nn.BatchNorm1d(self.emb_dim))
        fc2_dim = int(self.emb_dim/2)
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(self.emb_dim, fc2_dim), nn.ReLU(), nn.Linear(fc2_dim, output_dim))

    def forward_emb(self, x):
        for i, l in enumerate(self.fc_layers):
            x = self.fc_layers[i](x)
        x = self.emb_output(x)
        return x

    def forward_slab(self, x):
        x = self.output(x)
        x = F.softmax(x, 1)
        return x

    def forward(self, x):
        for i, l in enumerate(self.fc_layers):
            x = self.fc_layers[i](x)
        x = self.emb_output(x)
        x = self.output(x)
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
                # (batch_size, 2*hidden_dim)
        else:
            h = h[-1, 0, :, :]
        x = self.output(h)
        return x

    def forward_eval(self, x):
        '''x has shape (batch_size, seq_len, num_feats)'''
        o, (h, c) = self.seq_model(x, self.init_hidden(x.shape[0]))
        h = h.view(self.num_layers, self.num_directions, x.shape[0], self.hidden_dim)
        if self.bidirectional:
            h = torch.cat((h[-1, 0, :, :], h[-1, 1, :, :]), 1)
                # (batch_size, 2*hidden_dim)
        else:
            h = h[-1, 0, :, :]
        x = self.output(h)
        return x

class Attention_LSTM(nn.Module):
    '''bidirectional'''
    def __init__(self, input_dim, output_dim, args):
        super(Attention_LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_directions = 2

        self.seq_model = nn.LSTM(self.input_dim, self.hidden_dim, 
            self.num_layers, bidirectional=True, batch_first=True)

        self.output = nn.Linear(self.hidden_dim, self.output_dim)

        self.attention_w = nn.Linear(2*self.hidden_dim, self.hidden_dim)

    def attend(self, A):
        '''
        Args:
            A has shape (batch, seq_len, hidden_size*2)
        
        Return:
            Tensor with shape (batch, hidden_size)
        '''
        assert len(A.shape) == 3
        alpha = F.tanh(self.attention_w(A)) # (batch, seq_len, hidden_size)
        alpha = F.softmax(alpha, dim=-1)    # (batch, seq_len, hidden_size)
        beta = torch.sum(alpha, dim=1)      # (batch, hidden_size)
        return beta

    def forward(self, x):
        o, (h, c) = self.seq_model(x)
            # o shape: (seq_len, batch, hidden_size)
            # h shape: (num_layers, batch, hidden_size)
        weighted_repr = self.attend(o)
        x = self.output(weighted_repr)
        return x

    def forward_eval(self, x):
        o, (h, c) = self.seq_model(x)
        weighted_repr = self.attend(o)
        x = self.output(weighted_repr)
        return x

class CNN_LSTM(nn.Module):
    '''for wav data'''
    def __init__(self, input_dim, output_dim, args):
        super(CNN_LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.bidirectional = args.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.out_channels = args.out_channels
        self.window_size = args.window_size
        self.stride = args.stride

        self.conv1 = torch.nn.Sequential(
                        nn.Conv2d(self.input_dim, self.out_channels, 
                            (self.input_dim, self.window_size), stride=(1, self.stride)),
                        torch.nn.ReLU()
                    )
        self.seq_model = nn.LSTM(self.out_channels, self.hidden_dim, 
            self.num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.output = nn.Linear(self.hidden_dim*self.num_directions, self.output_dim)
    
    def init_hidden(self, batch_size):
        return(Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim)),
            Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim)))

    def forward(self, x):
        # todo pack_padded_sequence https://github.com/claravania/lstm-pytorch/blob/master/model.py
        x = x.view(x.shape[0], 1, 1, x.shape[1])
        x = self.conv1(x)
            # 5328, 80, 1, new_seq_len
        x = torch.squeeze(x)
            # 5328, 80, new_seq_len
        x = x.permute(0, 2, 1)
            # 5328, new_seq_len, 80
        o, (h, c) = self.seq_model(x, self.init_hidden(x.shape[0]))
        h = h.view(self.num_layers, self.num_directions, x.shape[0], self.hidden_dim)
        if self.bidirectional:
            h = torch.cat((h[-1, 0, :, :], h[-1, 1, :, :]), 1)
                # (batch_size, 2*hidden_dim)
        else:
            h = h[-1, 0, :, :]
        x = self.output(h)
        return x
    
    def forward_eval(self, x):
        # todo pack_padded_sequence https://github.com/claravania/lstm-pytorch/blob/master/model.py
        x = x.view(x.shape[0], 1, 1, x.shape[1])
        x = self.conv1(x)
            # 5328, 80, 1, new_seq_len
        x = torch.squeeze(x)
            # 5328, 80, new_seq_len
        x = x.permute(0, 2, 1)
            # 5328, new_seq_len, 80
        o, (h, c) = self.seq_model(x, self.init_hidden(x.shape[0]))
        h = h.view(self.num_layers, self.num_directions, x.shape[0], self.hidden_dim)
        if self.bidirectional:
            h = torch.cat((h[-1, 0, :, :], h[-1, 1, :, :]), 1)
                # (batch_size, 2*hidden_dim)
        else:
            h = h[-1, 0, :, :]
        x = self.output(h)
        return x

class CLSTM_MRK2(nn.Module):
    '''for wav data'''
    def __init__(self, input_dim, output_dim, args):
        super(CLSTM_MRK2, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim_1 = args.hidden_dim_1
        self.hidden_dim_2 = args.hidden_dim_2
        self.num_layers = args.num_layers
        self.bidirectional = args.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.out_channels = args.out_channels
        self.window_size = args.window_size
        self.stride = args.stride
        self.dropout = args.dropout
        self.lstm_dropout = 0.0

        num_out = int(math.log(self.out_channels/10, 2))+1
        out_list = [self.out_channels]
        curr_out = int(self.out_channels/2)
        while curr_out >= 10:
            out_list.append(curr_out)
            curr_out = int(curr_out/2)
        out_list.append(self.input_dim)
        out_list.reverse() # e.g. 1 -> 10 -> 20 -> 40 -> 80, num_out = 4

        window_list = [int(self.window_size/math.pow(2,i)) for i in range(num_out)]

        self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_list[i], out_list[i+1], 
                        (self.input_dim, window_list[i]), stride=(1, self.stride)),
                    nn.ReLU()
                )
            for i in range(num_out)])

        self.seq_model = nn.LSTM(self.out_channels, self.hidden_dim_1, 
            self.num_layers, bidirectional=self.bidirectional, dropout=self.lstm_dropout, batch_first=True)

        self.output = nn.Sequential(
                        nn.Linear(self.hidden_dim_1*self.num_directions, self.hidden_dim_2),
                        nn.Dropout(self.dropout),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim_2, self.output_dim)
                    )
    
    def init_hidden(self, batch_size):
        return(Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim_1)),
            Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim_1)))

    def forward(self, x):
        # todo pack_padded_sequence https://github.com/claravania/lstm-pytorch/blob/master/model.py
        x = x.view(x.shape[0], 1, 1, x.shape[1])
        for i, _ in enumerate(self.convs):
            x = self.convs[i](x)
            # 5328, 80, 1, new_seq_len
        x = torch.squeeze(x)
            # 5328, 80, new_seq_len
        x = x.permute(0, 2, 1)
            # 5328, new_seq_len, 80
        o, (h, c) = self.seq_model(x, self.init_hidden(x.shape[0]))
        h = h.view(self.num_layers, self.num_directions, x.shape[0], self.hidden_dim_1)
        if self.bidirectional:
            h = torch.cat((h[-1, 0, :, :], h[-1, 1, :, :]), 1)
                # (batch_size, 2*hidden_dim)
        else:
            h = h[-1, 0, :, :]
        x = self.output(h)
        return x
    
    def forward_eval(self, x):
        # todo pack_padded_sequence https://github.com/claravania/lstm-pytorch/blob/master/model.py
        x = x.view(x.shape[0], 1, 1, x.shape[1])
        for i, _ in enumerate(self.convs):
            x = self.convs[i](x)
            # 5328, 80, 1, new_seq_len
        x = torch.squeeze(x)
            # 5328, 80, new_seq_len
        x = x.permute(0, 2, 1)
            # 5328, new_seq_len, 80
        o, (h, c) = self.seq_model(x, self.init_hidden(x.shape[0]))
        h = h.view(self.num_layers, self.num_directions, x.shape[0], self.hidden_dim_1)
        if self.bidirectional:
            h = torch.cat((h[-1, 0, :, :], h[-1, 1, :, :]), 1)
                # (batch_size, 2*hidden_dim)
        else:
            h = h[-1, 0, :, :]
        x = self.output(h)
        return x

class SimpleSiamese(nn.Module):
    def __init__(self, input_dim, args):
        super(SimpleSiamese, self).__init__()
    
        self.input_dim = input_dim
        self.twin_dim = args.twin_dim

        mod = sys.modules[__name__]
        model_class = getattr(mod, args.twin)
        self.encoder = model_class(self.input_dim, self.twin_dim, args)

        self.join_dim = 4*self.twin_dim
        self.out = nn.Linear(self.join_dim, 2)

    def join(self, x1, x2):
        '''
        inputs have shape (batch_size, s1)
        returned shape: (batch_size, 4*s1)
        '''
        # return torch.cat([x1, x2],1)
        return torch.cat([x1, x2, torch.abs(x2-x1), x1*x2],1)

    def forward(self, x1, x2):
        '''returned shape: (batch_size, 2)'''
        o1 = self.encoder(x1)
        o2 = self.encoder(x2)

        join_output = self.join(o1, o2)
        pred = self.out(join_output)
        
        return pred

    def forward_x(self, x):
        return self.encoder(x)

    def forward_join(self, o1, o2):
        join_output = self.join(o1, o2)
        pred = self.out(join_output)
        return pred

    def forward_eval(self, x1, x2):
        '''returned shape: (batch_size, 2)'''
        o1 = self.encoder(x1)
        o2 = self.encoder(x2)

        join_output = self.join(o1, o2)
        pred = self.out(join_output)
        
        return pred