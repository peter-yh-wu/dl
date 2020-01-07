'''
Peter Wu
peterw1@andrew.cmu.edu
'''

import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.io import wavfile
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *

ggparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
src_dir = os.path.join(ggparent_dir, 'src')
sys.path.append(src_dir)
from global_utils import *

def get_rand_segments(x, seq_len):
    '''
    Args:
        x is a list of 1-dim np arrays
    return:
        np arr w shape (len(x), seq_len)
    '''
    all_data = []
    for data in x:
        start_index = np.random.randint(len(data)-seq_len+1)
        data = data[start_index:start_index+seq_len]
        all_data.append(data)
    xs = np.stack(all_data)
    return xs

def load_wav_files(paths):
    '''returns a list of 1-d numpy arrays'''
    all_data = []
    for p in paths:
        fs, data = wavfile.read(p) # data is 1-dim array
        all_data.append(data)
    return all_data

def load_wav_data(DATA_DIR):
    '''5564 train, 5328 dev, 5570 test'''
    WAV_DIR = os.path.join(DATA_DIR, 'wav')
    wav_files = os.listdir(WAV_DIR)
    wav_files = [f for f in wav_files if f.endswith('.wav')]

    train_files = [f for f in wav_files if f.startswith('train')]
    devel_files = [f for f in wav_files if f.startswith('devel')]
    test_files = [f for f in wav_files if f.startswith('test')]
    train_paths = [os.path.join(WAV_DIR, f) for f in train_files]
    devel_paths = [os.path.join(WAV_DIR, f) for f in devel_files]
    test_paths = [os.path.join(WAV_DIR, f) for f in test_files]

    train_x = load_wav_files(train_paths) # list of 1-dim np arrays
    devel_x = load_wav_files(devel_paths)

    return train_x, devel_x, train_files, devel_files

def load_soundnet_files(paths):
    x = []
    for p in paths:
        feats = np.load(p, encoding='latin1')
        feats = feats['arr_0']
        feats = np.mean(feats[4], 0) # shape: (512,)
        x.append(feats)
    x = np.stack(x)
    return x

def load_soundnet_data():
    DATA_DIR = '/home/srallaba/challenges/compare2019/sleepiness/feats'
    if not os.path.exists(DATA_DIR):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(parent_dir, 'data')
    DATA_DIR = os.path.join(DATA_DIR, 'soundnet')
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
    train_files = [f for f in files if f.startswith('train')]
    devel_files = [f for f in files if f.startswith('devel')]
    test_files = [f for f in files if f.startswith('test')]
    train_paths = [os.path.join(DATA_DIR, f) for f in train_files]
    devel_paths = [os.path.join(DATA_DIR, f) for f in devel_files]
    test_paths = [os.path.join(DATA_DIR, f) for f in test_files]
    train_x = load_soundnet_files(train_paths)
    dev_x = load_soundnet_files(devel_paths)
    return train_x, dev_x, train_files, devel_files

def load_mfcc_data():
    '''
    Return:
        train_x: list of (seq_len, num_feats)
        dev_x: list of (seq_len, num_feats)
        train_files: list of strings
        devel_files: list of strings
    '''
    DATA_DIR = '/home/srallaba/challenges/compare2019/sleepiness/feats/mfcc_kaldi'
    if not os.path.exists(DATA_DIR):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(parent_dir, 'data', 'mfcc')
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mfcc')]
    train_files = [f for f in files if f.startswith('train')]
    devel_files = [f for f in files if f.startswith('devel')]
    test_files = [f for f in files if f.startswith('test')]
    train_paths = [os.path.join(DATA_DIR, f) for f in train_files]
    devel_paths = [os.path.join(DATA_DIR, f) for f in devel_files]
    test_paths = [os.path.join(DATA_DIR, f) for f in test_files]
    train_x = [np.loadtxt(p) for p in train_paths]
    dev_x = [np.loadtxt(p) for p in devel_paths]
    return train_x, dev_x, train_files, devel_files

def load_baseline_data(feature_set, DATA_DIR, train_files=None, dev_files=None):
    task_name = 'ComParE2019_ContinuousSleepiness'
    feat_conf = {'ComParE':      (6373, 1, ';', 'infer'),
                'BoAW-125':     ( 250, 1, ';',  None),
                'BoAW-250':     ( 500, 1, ';',  None),
                'BoAW-500':     (1000, 1, ';',  None),
                'BoAW-1000':    (2000, 1, ';',  None),
                'BoAW-2000':    (4000, 1, ';',  None),
                'auDeep-40':    (1024, 2, ',', 'infer'),
                'auDeep-50':    (1024, 2, ',', 'infer'),
                'auDeep-60':    (1024, 2, ',', 'infer'),
                'auDeep-70':    (1024, 2, ',', 'infer'),
                'auDeep-fused': (4096, 2, ',', 'infer')}
    num_feat = feat_conf[feature_set][0]
    ind_off  = feat_conf[feature_set][1]
    sep      = feat_conf[feature_set][2]
    header   = feat_conf[feature_set][3]
    features_path = os.path.join(DATA_DIR, 'features') + '/'
    X_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
        # np array w/ shape (5564, 6373)
    X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
        # np array w/ shape (5328, 6373)
    if train_files is not None:
        x_train_indices = [int(f[6:-4])-1 for f in train_files] # filenames are 1-indexed
        X_train = X_train[x_train_indices, :]
    if dev_files is not None:
        x_dev_indices = [int(f[6:-4])-1 for f in dev_files] # filenames are 1-indexed
        X_devel = X_devel[x_dev_indices, :]
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_devel = scaler.transform(X_devel)
    return X_train, X_devel

def load_data(data_type, y_mode, process, pca_param=0.99):
    '''
    y-values returned are 0-indexed

    Args:
        data_type is a string or list, as is process
    
    Return:
        train_y: np array
        dev_y: np array
    '''
    DATA_DIR = '/home2/srallaba/challenges/compare2019/ComParE2019_ContinuousSleepiness'
    if not os.path.exists(DATA_DIR):
        DATA_DIR = '/home/srallaba/challenges/compare2019/sleepiness/data/ComParE2019_ContinuousSleepiness'
    if not os.path.exists(DATA_DIR):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(parent_dir, 'data')

    if isinstance(data_type, str):
        data_type = [data_type]
    if isinstance(process, str):
        process = [process]
    
    input_dim = 0
    if 'wav' in data_type:
        train_x, dev_x, _, _ = load_wav_data(DATA_DIR)
        input_dim += 1
    elif 'mfcc' in data_type:
        train_x, dev_x, _, _ = load_mfcc_data()
        input_dim += train_x[0].shape[1] # 39
    else:
        train_files = None
        dev_files = None
        train_x = None
        dev_x = None
        if 'soundnet' in data_type:
            train_x, dev_x, train_files, dev_files = load_soundnet_data()
            input_dim += 512
        baseline_feats = ['ComParE','BoAW-125','BoAW-250','BoAW-500','BoAW-1000','BoAW-2000','auDeep-40','auDeep-50','auDeep-60','auDeep-70','auDeep-fused']
        for feat in baseline_feats:
            if feat in data_type:
                curr_train_x, curr_dev_x = load_baseline_data(feat, DATA_DIR, train_files, dev_files)
                input_dim += curr_train_x.shape[1]
                if train_x is not None:
                    train_x = np.concatenate((train_x,curr_train_x), axis=1)
                else:
                    train_x = curr_train_x
                if dev_x is not None:
                    dev_x = np.concatenate((dev_x,curr_dev_x), axis=1)
                else:
                    dev_x = curr_dev_x

    label_file = os.path.join(DATA_DIR, 'lab', 'labels.csv')
    if not os.path.exists(label_file):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        label_file = os.path.join(parent_dir, 'data', 'labels.csv')
    df_labels = pd.read_csv(label_file)
    if 'soundnet' in data_type:
        train_files = [f[:-4]+'.wav' for f in train_files]
        dev_files = [f[:-4]+'.wav' for f in dev_files]
        train_y = pd.to_numeric(df_labels['label'][df_labels['file_name'].isin(train_files)]).values
        dev_y = pd.to_numeric(df_labels['label'][df_labels['file_name'].isin(dev_files)]).values
    else:
        train_y = pd.to_numeric( df_labels['label'][df_labels['file_name'].str.startswith('train')] ).values
        dev_y = pd.to_numeric( df_labels['label'][df_labels['file_name'].str.startswith('devel')] ).values

    output_dim = 9
    if y_mode == 'stark_3':
        train_indices = np.sort(np.concatenate([np.where(train_y == 1)[0], 
                            np.where(train_y == 5)[0], np.where(train_y == 9)[0]]))
        devel_indices = np.sort(np.concatenate([np.where(dev_y == 1)[0], 
                            np.where(dev_y == 5)[0], np.where(dev_y == 9)[0]]))
        
        if 'wav' in data_type or 'mfcc' in data_type:
            train_x = [train_x[i] for i in train_indices]
            dev_x = [dev_x[i] for i in devel_indices]
        else:
            train_x = train_x[train_indices, :]
            dev_x = dev_x[devel_indices, :]
        
        train_y = train_y[train_indices]
        dev_y = dev_y[devel_indices]
        train_y[train_y==5] = 2
        train_y[train_y==9] = 3
        dev_y[dev_y==5] = 2
        dev_y[dev_y==9] = 3
        output_dim = 3
    elif y_mode == 'no2':
        train_indices = np.where(train_y != 3)[0]
        devel_indices = np.where(dev_y != 3)[0]

        if 'wav' in data_type or 'mfcc' in data_type:
            train_x = [train_x[i] for i in train_indices]
            dev_x = [dev_x[i] for i in devel_indices]
        else:
            train_x = train_x[train_indices, :]
            dev_x = dev_x[devel_indices, :]

        train_y = train_y[train_indices]
        dev_y = dev_y[devel_indices]

        train_y[train_y>3] -= 1
        dev_y[dev_y>3] -= 1
        output_dim = 8

    if 'upsample' in process:
        y_values = list(set(np.unique(train_y)).union(set(np.unique(dev_y))))
        y_values.sort()
        y_indices_list = [np.where(train_y == i)[0] for i in y_values]
        y_value_counts = [len(y) for y in y_indices_list]
        max_ys = np.max(y_value_counts)
        new_y_value_counts = [max_ys-n for n in y_value_counts]
        new_bucket_indices = [np.random.choice(num_ys_prev, size=num_ys_new) 
                            for num_ys_new, num_ys_prev 
                                in zip(new_y_value_counts, y_value_counts)]
        new_x_indices = [y_indices_list[i][js] for i, js in enumerate(new_bucket_indices)]
            # list of 1-dim arrays, with each array comprised of indices of train_x
        new_x_indices = [item for sublist in new_x_indices for item in sublist]
        np.random.shuffle(new_x_indices)
        
        new_train_x = [train_x[i] for i in new_x_indices]
        if isinstance(train_x, list):
            train_x = train_x + new_train_x
        else:
            new_train_x = np.array(new_train_x)
            train_x = np.concatenate([train_x, new_train_x], axis=0)
        new_train_y = [train_y[i] for i in new_x_indices]
        new_train_y = np.array(new_train_y)
        train_y = np.concatenate([train_y, new_train_y])

    if 'pca' in process: # to-do consider whiten=True
        pca = PCA(n_components=pca_param)
        pca.fit(train_x) # shape: (n_samples, n_components)
        input_dim = pca.n_components_
        print('%d principle components' % input_dim)
        train_x = pca.transform(train_x)
        dev_x = pca.transform(dev_x)

    if isinstance(train_x, list):
        indices = [i for i in range(len(train_x))]
        zs = list(zip(train_x, indices))
        random.shuffle(zs)
        train_x, indices = zip(*zs)
        train_x = list(train_x)
        indices = list(indices)
        train_y = train_y[indices]
    else:
        p = np.random.permutation(len(train_x))
        train_x = train_x[p]
        train_y = train_y[p]
    return input_dim, output_dim, train_x, train_y-1, dev_x, dev_y-1

def predict(model, X_devel, pred_func=None): # todo add support for seq
    '''
    Args:
        pred_func: turns logits into y_pred

    Return:
        logits: shape (num_devel, num_classes)
        y_pred: numpy array of shape (num_devel,), elements are between
            0 and num_classes-1, inclusive
    '''
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X_devel)

        inputs = Variable(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        logits = model(inputs)
        if pred_func is None:
            values, indices = torch.max(logits, 1)
        else:
            _, indices = pred_func(logits)
        return logits, indices.numpy()

def train(model, optimizer, criterion, X_train, y_train, batch_size=64, optimizer2=None):
    '''
    Args:
        y_train: LongTensor of shape (num_train,)
    '''
    model.train()
    optimizer.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    l = 0
    num_samples = len(y_train)

    for i in range(0, num_samples, batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        x_lens = [len(x) for x in X_batch]
        if len(np.unique(x_lens)) > 1: # x_mode='truncate'
            min_len = np.min(x_lens)
            X_batch = np.array([x[:min_len] for x in X_batch])

        inputs = torch.FloatTensor(X_batch)
            # shape: 64, 79941

        inputs, targets = Variable(inputs), Variable(y_batch)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        if optimizer2 is not None:
            optimizer2.step()
        l += loss.item()

    return model

def get_metrics(y_true, y_pred):
    '''y values are 1-indexed'''
    spearman = spearmanr(y_true, y_pred)[0]
    if np.isnan(spearman):  # Might occur when the prediction is a constant
        spearman = 0.
    return accuracy_score(y_true, y_pred), \
        precision_score(y_true, y_pred, average='macro'), \
        recall_score(y_true, y_pred, average='macro'), \
        f1_score(y_true, y_pred, average='macro'), \
        confusion_matrix(y_true, y_pred), \
        spearman

def mk_y_slabs(ys, num_classes):
    '''Create soft labels based on given y-values
    
    Class y is mapped to [y/num_classes, 1-y/num_classes]

    Args:
        ys: tensor, y-values are 0-indexed
    '''
    col1 = ys.float()/num_classes
    col2 = 1-col1
    return torch.stack([col1, col2]).transpose(1,0)

def slab_predict(model, xs, num_classes): # todo add support for seq
    '''
    Return:
        logits
        preds
    '''
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(xs)

        inputs = Variable(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        logits = model(inputs)
        preds_tens = torch.clamp(torch.round(logits[:, 0]*num_classes), max=num_classes-1)
        
        return logits, preds_tens.cpu().numpy()

# -----------------------
# Functions unique to this experiment

def sample_batch(grouped_x, batch_size):
    '''
    Args:
        grouped_x: size-num_classes list of lists
    
    Return:
        x_tuple: size-9 list of size-batch_size lists
            (each sublist contains x values all having the same class,
            i.e. sublist 0 has y value 0)
    '''
    x_tuple = []
    for i, x_group in enumerate(grouped_x):
        curr_xs = random.sample(x_group, batch_size)
        x_tuple.append(curr_xs)
    return x_tuple

def mk_side_template():
    '''Template for when anchors are the side numbers

    Return:
        size-35 list of tuples (each elem in tuple is in {0,1,...8})
    '''
    return [
        (0,2,4),(0,2,5),(0,2,6),(0,2,7),(0,2,8),
        (0,3,5),(0,3,6),(0,3,7),(0,3,8),
        (0,4,6),(0,4,7),(0,4,8),(0,5,7),(0,5,8),(0,6,8),
        (1,3,5),(1,3,6),(1,3,7),(1,3,8),
        (1,4,6),(1,4,7),(1,4,8),(1,5,7),(1,5,8),(1,6,8),
        (2,4,6),(2,4,8),(2,4,8),(2,5,7),(2,5,8),(2,6,8),
        (3,5,7),(3,5,8),(3,6,8),(4,6,8)
    ]

def train_tup_slab(model, optimizer, criterion, x_tuple, y_true, template):
    '''Trains one batch of tuples
    
    Args:
        x_tuple: size-9 list of size_batch_size lists
            (sublists comprised of x_values having the same class)
        y_true: shape (9*batch_size,)
            [0,...0,1,...1,...8,...8] divided by something to make slab
        template: list of triples
            (triples are comprised of numbers in {0, 1, ..., 8})
    '''
    model.train()
    optimizer.zero_grad()
    batch_size = len(x_tuple[0])
    targets = Variable(y_true)
    if torch.cuda.is_available():
        targets = targets.cuda()
    o_list = []
    logits_list = []
    for i, curr_xs in enumerate(x_tuple):
        xs_tr = truncate_seqs(curr_xs)
        x_tens = torch.FloatTensor(xs_tr)
        x_var = Variable(x_tens) # shape: (batch_size, seq_len)
        if torch.cuda.is_available():
            x_var = x_var.cuda()
        o = model.forward_emb(x_var) # shape: (batch_size, emb_dim)
        o_list.append(o)
        logits = model.forward_slab(o) # shape: (batch_size, 2)
        logits_list.append(logits)
    o_tens = torch.stack(o_list) # shape: (9, batch_size, emb_dim)
    logits_stack = torch.stack(logits_list) # shape: (9, batch_size, 2)
    logits_stack = logits_stack.view(-1, 9) # shape: (9*batch_size, 2)
    l = criterion(logits_stack, targets, o_tens, template)
    l.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
    return model

def logistic(x):
    '''Logistic Function: log_2 (1+2^{-x})'''
    # return torch.log2(1+torch.pow(2, x))
    return torch.log(1+torch.exp(x))

class FRLoss(nn.Module):
    def __init__(self):
        super(FRLoss, self).__init__()
        self.pdist  = PairwiseDistance(2)
    
    def forward(self, anchor, positive, negative):
        pos_dist   = self.pdist.forward(anchor, positive)
        neg_dist   = self.pdist.forward(anchor, negative)
        return torch.mean(logistic(neg_dist-pos_dist))

class TupleLoss(nn.Module):
    def __init__(self):
        super(TupleLoss, self).__init__()
        self.fr_loss = FRLoss()
    
    def forward(self, all_o, template):
        '''
        Args:
            all_o: shape (9, batch_size, emb_dim)
            template: list of triples
                (triples are comprised of numbers in {0, 1, ..., 8})
        Return:
            average of all triplet losses
        '''
        tot_l = torch.tensor(0.0)
        for (l, m, r) in template: # anchor is left
            tot_l += self.fr_loss(all_o[l, :, :], all_o[m, :, :], all_o[r, :, :])
        for (l, m, r) in template: # anchor is right
            tot_l += self.fr_loss(all_o[r, :, :], all_o[m, :, :], all_o[l, :, :])
        return tot_l / 2 / len(template)

class TupleSlabLoss(nn.Module):
    def __init__(self, alpha=0.1, weight=None):
        super(TupleSlabLoss, self).__init__()
        self.tuple_loss = TupleLoss()
        self.cross_entropy = nn.BCELoss(weight=weight)
        self.alpha = alpha

    def forward(self, all_logits, y_true, all_o, template):
        '''
        Args:
            all_logits: shape (9*batch_size, 2)
            y_true: shape (9*batch_size, 2)
                = TODO
            all_o: shape (9, batch_size, emb_dim)
            template: list of triples
                (triples are comprised of numbers in {0, 1, ..., 8})
        '''
        loss1 = self.cross_entropy(all_logits, y_true)
        loss2 = self.tuple_loss(all_o, template)
        loss  = loss1+self.alpha*loss2
        return loss
