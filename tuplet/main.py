'''Tuplet loss implementation

Peter Wu
peterw1@andrew.cmu.edu
'''

import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
import torch

import model

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from model_utils import *

ggparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
src_dir = os.path.join(ggparent_dir, 'src')
sys.path.append(src_dir)
from logger import *
from global_utils import *
from global_sia_utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_a', help='number of a values per batch;', type=int, default=32)
    parser.add_argument('-num_n', help='number of n values for each a value;', type=int, default=32)
    parser.add_argument('-p_margin', help='p margin;', type=int, default=1)
    parser.add_argument('-n_margin', help='n margin;', type=int, default=2)
    parser.add_argument('-num_batches', help='number of batches per epoch;', type=int, default=100)
    parser.add_argument('-num_epochs', help='number of epochs;', type=int, default=100)
    parser.add_argument('-batch_size', help='batch size;', type=int, default=64)
    parser.add_argument('-data', help='wav, mfcc, soundnet, ComParE, ...;', nargs='+', type=str, default='ComParE')
    parser.add_argument('-process', help='process data: none, upsample, pca, ...;', nargs='+', type=str, default='none')
    parser.add_argument('-segment', help='whether to segment datapoints;', type=bool, default=False)
    parser.add_argument('-y_mode', help='stark_3, no2, or normal;', type=str, default="normal")
    parser.add_argument('-pca_param', help='fraction of variance retained;', type=float, default=0.99)
    parser.add_argument('-lr', help='lr;', type=float, default=1e-05)
    parser.add_argument('-dropout', help='dropout;', type=float, default=0.0)
    parser.add_argument('-num_layers', help='number of layers;', type=int, default=3)
    parser.add_argument('-hidden_dim', help='hidden_dim;', type=int, default=128)
    parser.add_argument('-hidden_dim_1', help='hidden_dim_1;', type=int, default=128)
    parser.add_argument('-hidden_dim_2', help='hidden_dim_2;', type=int, default=32)
    parser.add_argument('-loss_dim', help='for all-threshold loss;', type=int, default=10)
    parser.add_argument('-out_channels', help='out_channels;', type=int, default=80)
    parser.add_argument('-bidirectional', help='bidirectional;', type=bool, default=True)
    parser.add_argument('-seq_len', help='seq_len;', type=int, default=1200) 
    parser.add_argument('-window_size', help='window_size;', type=int, default=400) 
    parser.add_argument('-stride', help='stride;', type=int, default=5) 
    parser.add_argument('-model', help='model;', type=str, default="CNN_LSTM")
    parser.add_argument('-model_type', help='funnel, triangle, or block;', type=str, default="funnel")
    parser.add_argument('-softmax', help='softmax;', type=bool, default=False)
    parser.add_argument('-twin', help='twin;', type=str, default="Simple_LSTM")
    parser.add_argument('-twin_dim', help='twin net output dim;', type=int, default=10)
    parser.add_argument('-sia_mode', help='vote or _sort;', type=str, default="vote")
    parser.add_argument('-max_sia_dev', help='max sia dev samples;', type=int, default=5000)
    parser.add_argument('-num_quick', help='number of quicksorts;', type=int, default=4)
    parser.add_argument('-loss_type', help='wloss, focal, slab, bin, all_thres, ordistic, or normal;', type=str, default="normal")
    parser.add_argument('-loss_param', help='e.g. gamma value for focal;', type=str, default="2.0")
    parser.add_argument('-log_path', help='log path;', type=str, default="log")
    parser.add_argument('-debug', help='whether to use debug mode;', type=bool, default=False)
    parser.add_argument('-patience', help='patience;', type=int, default=5)
    parser.add_argument('-seed', help='random seed;', type=int, default=0)

    return parser.parse_args()

def main(verbose=True):
    args = parse_args()
    data_type = args.data
    if isinstance(data_type, str):
        data_type = [data_type]

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dim, output_dim, train_x, train_y, dev_x, dev_y = \
        load_data(args.data, args.y_mode, args.process, args.pca_param)
    y_values = list(set(np.unique(train_y)).union(set(np.unique(dev_y))))
    y_values.sort()
    
    output_dim = 2
    
    if 'wav' in data_type or 'mfcc' in data_type:
        dev_x = truncate_seqs(dev_x)
    if args.debug:
        print('done loading data')

    y_indices_list_train = [np.where(train_y == i)[0] for i in y_values]
    y_value_counts_train = [len(y) for y in y_indices_list_train]
    y_indices_list_dev = [np.where(dev_y == i)[0] for i in y_values]
    y_value_counts_dev = [len(y) for y in y_indices_list_dev]
    y_value_counts = [t+d for t, d in zip(y_value_counts_train, y_value_counts_dev)]

    num_classes = len(y_values)
    grouped_train_x = group_data_by_class(train_x, train_y, num_classes)

    grouped_train_x_flat = [item for sublist in grouped_train_x for item in sublist]
    grouped_train_y = [[i]*count for i, count in enumerate(y_value_counts_train)]
    grouped_train_y_flat = [item for sublist in grouped_train_y for item in sublist]
    grouped_train_y_start_is = [y_value_counts_train[0]]
    for num_y in y_value_counts_train[1:]:
        grouped_train_y_start_is.append(num_y+grouped_train_y_start_is[-1])

    train_y_tensor = torch.LongTensor(train_y) # y-values 0-indexed
    dev_y_tensor = torch.LongTensor(dev_y)
    
    train_y_tensor = mk_y_slabs(train_y_tensor, num_classes)
        # shape: (num_train, 2)
    dev_y_tensor = mk_y_slabs(dev_y_tensor, num_classes)
    if torch.cuda.is_available():
        train_y_tensor = train_y_tensor.cuda()
        dev_y_tensor = dev_y_tensor.cuda()
    assert torch.min((train_y_tensor)).item() >= 0
    assert torch.min((dev_y_tensor)).item() >= 0

    loss_weights = None
    num_train = len(train_y_tensor)
    loss_weights = num_train/2/torch.sum(train_y_tensor, 0)
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()

    y_tuple = mk_y_slabs(torch.arange(9).repeat(args.batch_size,1).transpose(0,1).reshape(-1), num_classes)
    template = mk_side_template()

    model_class = getattr(model, args.model)
    net = model_class(input_dim, output_dim, args)
    
    print_log(net, args.log_path)
    if torch.cuda.is_available():
        net.cuda()
    
    criterion = TupleSlabLoss(alpha=float(args.loss_param), weight=loss_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    _, y_pred = slab_predict(net, dev_x, num_classes)
    
    acc, precision, recall, f1, conf_mat, spearman = get_metrics(dev_y, y_pred)
    best_acc = acc
    best_met = spearman
    print_log('acc before training: %f' % best_acc, args.log_path)
    print_log('spearman before training: %f' % best_met, args.log_path)
    
    best_val_loss = sys.maxsize
    prev_best_epoch = 0
    for e in range(args.num_epochs):
        for b in range(args.num_batches):
            x_tuple = sample_batch(grouped_train_x, args.batch_size)
            net = train_tup_slab(net, optimizer, criterion, x_tuple, y_tuple, template)
            
        logits, y_pred = slab_predict(net, dev_x, num_classes)
        loss = criterion.cross_entropy(logits, dev_y_tensor)
    
        val_loss = loss.item()
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            prev_best_epoch = e
        elif e - prev_best_epoch > args.patience:
            break
        acc, precision, recall, f1, conf_mat, spearman = get_metrics(dev_y, y_pred)
        print_log('%f, %f, %f, %f, %f' % (acc, precision, recall, f1, spearman), args.log_path)
        
        if acc > best_acc:
            best_acc = acc
        if spearman > best_met:
            best_met = spearman
    print_log('%f %f' % (best_acc, best_met), args.log_path)
    
    model_path = 'model.ckpt'
    conf_mat_path = 'conf_mat.npy'
    sia_conf_mat_path = 'sia_conf_mat.npy'
    slash_index = args.log_path.rfind('/')
    dot_index = args.log_path.rfind('.')
    if slash_index == -1:
        if dot_index != -1:
            model_path = args.log_path[:dot_index]+'.ckpt'
            conf_mat_path = args.log_path[:dot_index]+'-conf_mat.npy'
            sia_conf_mat_path = args.log_path[:dot_index]+'-sia_conf_mat.npy'
    else:
        if dot_index == -1 or dot_index < slash_index:
            model_path = args.log_path[:slash_index]+'/model.ckpt'
            conf_mat_path = args.log_path[:slash_index]+'/conf_mat.npy'
            sia_conf_mat_path = args.log_path[:slash_index]+'/sia_conf_mat.npy'
        else:
            model_path = args.log_path[:dot_index]+'.ckpt'
            conf_mat_path = args.log_path[:dot_index]+'-conf_mat.npy'
            sia_conf_mat_path = args.log_path[:dot_index]+'-sia_conf_mat.npy'
    if args.model == 'forest':
        joblib.dump(net, model_path)
    else:
        torch.save(net.state_dict(), model_path)
    np.save(conf_mat_path, conf_mat)
    if args.model == 'Siamese':
        np.save(sia_conf_mat_path, sia_conf_mat)

if __name__ == "__main__":
    main()