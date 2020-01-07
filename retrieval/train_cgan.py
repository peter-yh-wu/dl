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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm

from data_utils3 import mk_dataloader, load_pkl
from cgan3 import Generator, Discriminator


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
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--lr', default=1e-6, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int, help='Number of epochs to update the learning rate.')
    parser.add_argument('--log_step', default=1000, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='runs/runX', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--text_i', default=10, type=int, help='text index to synthesize')
    args = parser.parse_args()

    FloatTensor = torch.cuda.FloatTensor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_loader = mk_dataloader('train', batch_size=args.batch_size, audio=args.audio)
    val_loader = mk_dataloader('dev', batch_size=args.batch_size, shuffle=False, audio=args.audio)
    test_loader = mk_dataloader('test', batch_size=args.batch_size, shuffle=False, audio=args.audio)

    text_file_path = 'text/train.txt'
    with open(text_file_path, 'r') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    l_list = [l.split('\t') for l in lines]
    ids = [l[0] for l in l_list]
    texts = [l[1] for l in l_list]

    hidden_dim = 128
    cond_emb_dim = 768

    G = Generator(hidden_dim=hidden_dim, cond_emb_dim=cond_emb_dim)
    D = Discriminator(cond_emb_dim=cond_emb_dim)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G = G.cuda(args.cuda)
    D = D.cuda(args.cuda)

    G_ckpt_path = 'G_%d.ckpt' % args.cuda
    D_ckpt_path = 'D_%d.ckpt' % args.cuda
    if os.path.exists(G_ckpt_path):
        G.load_state_dict(torch.load(G_ckpt_path))
    if os.path.exists(D_ckpt_path):
        D.load_state_dict(torch.load(D_ckpt_path))
    
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr/1e2, betas=(0.5, 0.999))
    
    BCE_loss = nn.BCELoss()

    best_loss = float("inf")
    plt.figure(figsize=(10,10))

    text_i = args.text_i

    for mel1, mel2, text_emb, idx_batch, idx_batch2 in val_loader:
        mel1 = mel1.cuda(args.cuda)
        mel2 = mel2.cuda(args.cuda)
        orig = mel1[text_i].cpu().numpy()
        target = mel2[text_i].cpu().numpy()
        idx = idx_batch[text_i].numpy()
        text = texts[idx]
        idx2 = idx_batch2[text_i].numpy()
        new_text = texts[idx2]
        with open('text.txt', 'w+') as ouf:
            ouf.write('orig: %s\n   ' % text)
            ouf.write('new: %s\n' % new_text)
        print('orig: %s' % text)
        print('new: %s' % new_text)
        np.save('mel_orig.npy', orig.T)
        plt.imshow(orig.T)
        plt.savefig('mel_orig.png')
        plt.clf()
        np.save('mel_target.npy', target.T)
        plt.imshow(target.T)
        plt.savefig('mel_target.png')
        plt.clf()
        break

    print("epoch\tbatch_i\tg_mse\tg_bce\td_real\td_fake\td_fake2")

    for epoch in range(args.epochs):
        G.train()
        D.train()
        train_losses = []
        for batch_idx, (mel1, mel2, text_emb, idx_batch, idx_batch2) in enumerate(train_loader):
            mel1 = mel1.cuda(args.cuda) # orig
            mel2 = mel2.cuda(args.cuda) # new
            text_emb = text_emb.cuda(args.cuda)

            valid = Variable(FloatTensor(mel1.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(mel1.shape[0], 1).fill_(0.0), requires_grad=False)

            # -------

            G_optimizer.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0, 1, (mel1.shape[0], hidden_dim, 1, 1))))
            text_emb_g = text_emb.unsqueeze(2).unsqueeze(3)
            gen_imgs = G(z, text_emb_g, mel1.unsqueeze(1))

            text_emb = text_emb.unsqueeze(2).unsqueeze(3).expand(text_emb.shape[0],768,128,64)
            validity = D(gen_imgs, text_emb)
            g_mse_loss = F.mse_loss(gen_imgs, mel2.unsqueeze(1))
            g_bce_loss = BCE_loss(validity, valid)
            g_loss = g_mse_loss + g_bce_loss

            g_loss.backward()
            G_optimizer.step()

            # -------

            D_optimizer.zero_grad()

            validity_real = D(mel2.unsqueeze(1), text_emb)
            d_real_loss = BCE_loss(validity_real, valid)

            validity_fake = D(gen_imgs.detach(), text_emb)
            d_fake_loss = BCE_loss(validity_fake, fake)

            validity = D(mel1.unsqueeze(1), text_emb)
            d_fake_loss2 = BCE_loss(validity, fake)

            d_loss = (d_real_loss + d_fake_loss + d_fake_loss2) / 3
            d_loss.backward()
            D_optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    "%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f"
                    % (epoch, batch_idx, g_mse_loss.item(), g_bce_loss.item(), 
                        d_real_loss.item(), d_fake_loss.item(), d_fake_loss2.item())
                )

            if batch_idx % 300 == 0:
                with torch.no_grad():
                    for mel1, mel2, text_emb, idx_batch, idx_batch2 in val_loader:
                        mel1 = mel1.cuda(args.cuda)
                        mel2 = mel2.cuda(args.cuda)
                        text_emb = text_emb.cuda(args.cuda)
                        orig = mel1[text_i].cpu().numpy()
                        target = mel2[text_i].cpu().numpy()

                        z = Variable(FloatTensor(np.random.normal(0, 1, (mel1.shape[0], hidden_dim, 1, 1))))
                        text_emb_g = text_emb.unsqueeze(2).unsqueeze(3)
                        gen_imgs = G(z, text_emb_g, mel1.unsqueeze(1))

                        np_img = gen_imgs[text_i, 0, :, :].detach().cpu().numpy()
                        np.save('mel_%d_%d.npy' % (epoch, batch_idx), np_img.T)
                        plt.imshow(np_img.T)
                        plt.savefig('mel_%d_%d.png' % (epoch, batch_idx))
                        plt.clf()
                        break


if __name__ == '__main__':
    main()
