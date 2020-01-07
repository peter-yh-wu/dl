'''Classifer using loss function built for Spearman's rank correlation coefficient

Loss function in model_utils.py

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

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from model_utils import *

ggparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
src_dir = os.path.join(ggparent_dir, 'src')
sys.path.append(src_dir)
from logger import *
from global_utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-softmax', help='whether to use softmax;', type=bool, default=False)
    parser.add_argument('-batch_size', help='batch size;', type=int, default=64)
    parser.add_argument('-data', help='wav, mfcc, ComParE, ...;', nargs='+', type=str, default='ComParE')
    parser.add_argument('-process', help='process data: none, upsample, pca, ...;', nargs='+', type=str, default='none')
    parser.add_argument('-segment', help='whether to segment datapoints;', type=bool, default=False)
    parser.add_argument('-y_mode', help='modifications to y-values;', type=str, default="normal")
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
    parser.add_argument('-model', help='model;', type=str, default="MLP")
    parser.add_argument('-model_type', help='funnel, triangle, or block;', type=str, default="funnel")
    parser.add_argument('-loss_type', help='criterion;', type=str, default="normal")
    parser.add_argument('-loss_param', help='e.g. gamma value for focal;', type=str, default="2.0")
    parser.add_argument('-log_path', help='log path;', type=str, default="log")
    parser.add_argument('-debug', help='whether to use debug mode;', type=bool, default=False)
    parser.add_argument('-num_epochs', help='number of epochs;', type=int, default=100)
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

    input_dim, output_dim, train_x, train_y, dev_x, dev_y = \
        load_data(args.data, args.y_mode, args.process, args.pca_param)
    y_values = list(set(np.unique(train_y)).union(set(np.unique(dev_y))))
    y_values.sort()
    if 'wav' in data_type or 'mfcc' in data_type:
        dev_x = truncate_seqs(dev_x)
    if args.debug:
        print('done loading data')

    train_y_tensor = torch.LongTensor(train_y) # y-values 0-indexed
    dev_y_tensor = torch.LongTensor(dev_y) # y-values 0-indexed
    if torch.cuda.is_available():
        train_y_tensor = train_y_tensor.cuda()
        dev_y_tensor = dev_y_tensor.cuda()
    num_classes = len(y_values)

    y_indices_list_train = [np.where(train_y == i)[0] for i in y_values]
    y_value_counts_train = [len(y) for y in y_indices_list_train]
    y_indices_list_dev = [np.where(dev_y == i)[0] for i in y_values]
    y_value_counts_dev = [len(y) for y in y_indices_list_dev]
    y_value_counts = [t+d for t, d in zip(y_value_counts_train, y_value_counts_dev)]

    loss_weights = None
    if args.loss_type == 'wloss':
        min_count = np.min(y_value_counts)
        weight_list = [min_count/c for c in y_value_counts]
        loss_weights = torch.FloatTensor(weight_list)

    model_class = getattr(model, args.model)
    net = model_class(input_dim, output_dim, args)
    print_log(net, args.log_path)
    if torch.cuda.is_available():
        net.cuda()

    criterion = SpearLoss2()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    _, y_pred = predict(net, dev_x)
    acc, precision, recall, f1, conf_mat, spearman = get_metrics(dev_y, y_pred)

    best_acc = acc
    best_met = spearman
    print_log('acc before training: %f' % best_acc, args.log_path)
    print_log('spearman before training: %f' % best_met, args.log_path)

    best_val_loss = sys.maxsize
    prev_best_epoch = 0
    for e in range(args.num_epochs):
        if 'wav' in data_type and args.segment:
            train_xs = get_rand_segments(train_x, args.seq_len)
            net = train(net, optimizer, criterion, train_xs, train_y_tensor, batch_size=args.batch_size)
        else:
            net = train(net, optimizer, criterion, train_x, train_y_tensor, batch_size=args.batch_size)
        logits, y_pred = predict(net, dev_x)
            # logits shape: (num_devel, num_classes)
        if args.debug:
            print("epoch %d:" % (e+1), logits)
        loss = criterion(logits, dev_y_tensor)
        
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
            torch.save(net.state_dict(), model_path)
            np.save(conf_mat_path, conf_mat)

        print_log('%f %f' % (best_acc, best_met), args.log_path)

    if args.model == 'forest' or args.model == 'svm':
        joblib.dump(net, model_path)

if __name__ == "__main__":
    main()