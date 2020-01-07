'''
Peter Wu
peterw1@andrew.cmu.edu
'''

import math
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

def load_mfcc_data(DATA_DIR):
    '''
    Return:
        train_x: list of (seq_len, num_feats)
        dev_x: list of (seq_len, num_feats)
        train_files: list of strings
        devel_files: list of strings
    '''
    mfcc_dir = os.path.join(DATA_DIR, 'mfcc')
    files = [f for f in os.listdir(mfcc_dir) if f.endswith('.mfcc')]
    train_files = [f for f in files if f.startswith('train')]
    devel_files = [f for f in files if f.startswith('devel')]
    test_files = [f for f in files if f.startswith('test')]
    train_paths = [os.path.join(mfcc_dir, f) for f in train_files]
    devel_paths = [os.path.join(mfcc_dir, f) for f in devel_files]
    test_paths = [os.path.join(mfcc_dir, f) for f in test_files]
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
        train_x, dev_x, _, _ = load_mfcc_data(DATA_DIR)
        input_dim += train_x[0].shape[1]
    else:
        train_files = None
        dev_files = None
        train_x = None
        dev_x = None
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
    train_y = pd.to_numeric( df_labels['label'][df_labels['file_name'].str.startswith('train')] ).values
    dev_y = pd.to_numeric( df_labels['label'][df_labels['file_name'].str.startswith('devel')] ).values

    output_dim = 9

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

    if 'pca' in process:
        pca = PCA(n_components=pca_param)
        pca.fit(train_x) # shape: (n_samples, n_components)
        input_dim = pca.n_components_
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

def predict(model, X_devel):
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
        values, indices = torch.max(logits, 1)
        return logits, indices.cpu().numpy()

def train(model, optimizer, criterion, X_train, y_train, batch_size=64):
    '''
    Args:
        y_train: LongTensor of shape (num_train,)
    '''
    model.train()
    optimizer.zero_grad()
    l = 0
    num_samples = len(y_train)

    for i in range(0, num_samples, batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        x_lens = [len(x) for x in X_batch]
        if len(np.unique(x_lens)) > 1:
            min_len = np.min(x_lens)
            X_batch = np.array([x[:min_len] for x in X_batch])

        inputs = torch.FloatTensor(X_batch)

        inputs, targets = Variable(inputs), Variable(y_batch)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        l += loss.item()

    return model

def get_metrics(y_true, y_pred):
    '''y values are 1-indexed'''
    spearman = spearmanr(y_true, y_pred)[0]
    if np.isnan(spearman):
        spearman = 0.
    return accuracy_score(y_true, y_pred), \
        precision_score(y_true, y_pred, average='macro'), \
        recall_score(y_true, y_pred, average='macro'), \
        f1_score(y_true, y_pred, average='macro'), \
        confusion_matrix(y_true, y_pred), \
        spearman

class SpearLoss(nn.Module):
    """
    1 - spearman
    """

    def __init__(self):
        super(SpearLoss, self).__init__()

    def forward(self, logits, y_true):
        '''
        Args:
            logits: tensor w/ shape (batch_size, num_classes)
            y_true: tensor w/ shape (batch_size,)
        '''
        y_pred_np = torch.max(logits, 1)[1].cpu().numpy()
        y_true_np = y_true.numpy()
        spear = spearmanr(y_true_np, y_pred_np)[0]
        loss_np = 1-spear
        loss = torch.tensor(loss_np, requires_grad=True)
        return loss*torch.log(logits).mean()

class SpearLoss2(nn.Module):
    """
    exp(1-spearman) - 1
    """

    def __init__(self):
        super(SpearLoss2, self).__init__()

    def forward(self, logits, y_true):
        '''
        Args:
            logits: tensor w/ shape (batch_size, num_classes)
            y_true: tensor w/ shape (batch_size,)
        '''
        y_pred_np = torch.max(logits, 1)[1].cpu().numpy()
        y_true_np = y_true.numpy()
        spear = spearmanr(y_true_np, y_pred_np)[0]
        loss_np = math.exp(1-spear)-1
        loss = torch.tensor(loss_np, requires_grad=True)
        return loss*torch.log(logits).mean()
