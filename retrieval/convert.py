'''
Peter Wu
peterw1@andrew.cmu.edu
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import logging
import numpy as np
import os
import pickle
import random
import shutil
import sys
import tensorboard_logger as tb_logger
import time
import torch

from torch.nn.utils.clip_grad import clip_grad_norm

from data_utils import mk_convert_dataloader, load_pkl
from models import CNNEncDec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs.')
    parser.add_argument('--hidden_dim', default=256, type=int, help='LSTM hidden dim for Mel encoder')
    parser.add_argument('--emb_dim', default=256, type=int, help='embedding dimension')
    parser.add_argument('--num_layers', default=3, type=int, help='Number of LSTM layers for Mel encoder')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('--audio', default='mel', type=str, help='mel or cqt')
    parser.add_argument('--enc2', default='cnn', type=str, help='lstm or cnn')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--lr', default=.0002, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int, help='Number of epochs to update the learning rate.')
    parser.add_argument('--log_step', default=1000, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='runs/runX', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_loader = mk_convert_dataloader('train', batch_size=args.batch_size, audio=args.audio)
    val_loader = mk_convert_dataloader('dev', batch_size=args.batch_size, shuffle=False, audio=args.audio)
    test_loader = mk_convert_dataloader('test', batch_size=args.batch_size, shuffle=False, audio=args.audio)

    model = CNNEncDec()
    model = model.cuda(args.cuda)
    ckpt_path = 'device%d.ckpt' % args.cuda
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    plt.figure(figsize=(10,10))

    for mel1, mel2 in val_loader:
        mel1 = mel1.cuda(args.cuda)
        mel2 = mel2.cuda(args.cuda)
        orig = mel1[0].cpu().numpy()
        target = mel2[0].cpu().numpy()
        plt.imshow(orig.T)
        plt.savefig('orig.png')
        plt.clf()
        plt.imshow(target.T)
        plt.savefig('target.png')
        plt.clf()
        break

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch_idx, (mel1, mel2) in enumerate(train_loader):
            mel1 = mel1.cuda(args.cuda)
            mel2 = mel2.cuda(args.cuda)
            melp = model(mel1)
            loss = loss_fn(melp, mel2)
            train_losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        with torch.no_grad():
            model.eval()
            val_losses  = []
            for mel1, mel2 in val_loader:
                mel1 = mel1.cuda(args.cuda)
                mel2 = mel2.cuda(args.cuda)
                melp = model(mel1)
                loss = loss_fn(melp, mel2)
                val_losses.append(loss.item())
            for mel1, mel2 in val_loader:
                mel1 = mel1.cuda(args.cuda)
                mel2 = mel2.cuda(args.cuda)
                melp = model(mel1)
                orig = mel1[0].cpu().numpy()
                target = mel2[0].cpu().numpy()
                new = melp[0].cpu().numpy()
                plt.imshow(target.T)
                plt.savefig('target.png')
                plt.clf()
                plt.imshow(new.T)
                plt.savefig('new_%d.png' % epoch)
                plt.clf()
                break
        
        val_loss = np.mean(val_losses)
        if val_loss < best_loss:
            torch.save(model.state_dict(), ckpt_path)
        print(np.mean(train_losses), val_loss)


if __name__ == '__main__':
    main()
