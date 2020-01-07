'''
Peter Wu
peterw1@andrew.cmu.edu
'''

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

from data_utils import mk_dataloader, load_pkl
from models import SimpleDeviseModel, ContrastiveLoss
from utils import i2t


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
    parser.add_argument('--enc2', default='lstm', type=str, help='lstm or cnn')
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

    train_loader = mk_dataloader('train', batch_size=args.batch_size, audio=args.audio)
    val_retrieval_loader = mk_dataloader('dev', retrieval=True, batch_size=args.batch_size, audio=args.audio)
    test_retrieval_loader = mk_dataloader('test', retrieval=True, batch_size=args.batch_size, audio=args.audio)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(args.logger_name, flush_secs=5)

    model_hidden_size = args.hidden_dim
    model_embedding_size = args.emb_dim
    model_num_layers = args.num_layers
    if args.audio == 'mel':
        audio_dim = 80
    else:
        audio_dim = 84
    model = SimpleDeviseModel(audio_dim, args.enc2, model_hidden_size, model_embedding_size, model_num_layers)
    model = model.cuda(args.cuda)
    ckpt_path = 'device%d.ckpt' % args.cuda
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = ContrastiveLoss(margin=args.margin, max_violation=args.max_violation, cuda=args.cuda)

    # Train the Model
    Eiters = 0
    best_rsum = 0
    best_r1 = 0
    best_r5 = 0
    best_r10 = 0
    best_r50 = 0
    best_r100 = 0
    best_medr = sys.maxsize
    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        model.train()
        # --- train --------------------------
        train_losses = []
        for batch_idx, (train_text_embs, train_spkr_embs) in enumerate(train_loader):
            Eiters += 1
            train_text_embs = train_text_embs.cuda(args.cuda)
            train_spkr_embs = train_spkr_embs.cuda(args.cuda)
            o1, o2 = model.forward(train_text_embs, train_spkr_embs)
            loss = loss_fn(o1, o2)
            train_losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        print('Epoch %d' % epoch)

        # --- val ----------------------------
        with torch.no_grad():
            model.eval()
            val_losses  = [] # devise losses
            all_transformed_text_embs = []
            all_transformed_mel_embs = []
            for val_text_embs, val_spkr_embs in val_retrieval_loader:
                val_text_embs = val_text_embs.cuda(args.cuda)
                val_spkr_embs = val_spkr_embs.cuda(args.cuda)
                o1, o2 = model.forward(val_text_embs, val_spkr_embs)
                loss = loss_fn(o1, o2)
                val_losses.append(loss.item())
                all_transformed_text_embs.append(o1.cpu().numpy())
                all_transformed_mel_embs.append(o2.cpu().numpy())
            all_transformed_text_embs = np.concatenate(all_transformed_text_embs, axis=0)
            all_transformed_mel_embs = np.concatenate(all_transformed_mel_embs, axis=0)

            (r1i, r5i, r10i, r50, r100, medri) = i2t(all_transformed_mel_embs, all_transformed_text_embs)
            logging.info("Val: %.1f%%, %.1f%%, %.1f%%, %.1f%%, %.1f%%, %d" %
                        (r1i, r5i, r10i, r50, r100, int(medri)))
            # sum of recalls to be used for early stopping
            currscore = r1i + r5i + r10i
            if r1i > best_r1:
                best_r1 = r1i
            if r5i > best_r5:
                best_r5 = r5i
            if r10i > best_r10:
                best_r10 = r10i
            if r50 > best_r50:
                best_r50 = r50
            if r100 > best_r100:
                best_r100 = r100
            if medri < best_medr:
                best_medr = medri
            logging.info("Best Val: %.1f%%, %.1f%%, %.1f%%, %.1f%%, %.1f%%, %d" %
                        (best_r1, best_r5, best_r10, best_r50, best_r100, int(best_medr)))

            # record metrics in tensorboard
            tb_logger.log_value('r1i', r1i, step=Eiters)
            tb_logger.log_value('r5i', r5i, step=Eiters)
            tb_logger.log_value('r10i', r10i, step=Eiters)
            tb_logger.log_value('medri', medri, step=Eiters)
            tb_logger.log_value('rsum', currscore, step=Eiters)

            # remember best R@ sum and save checkpoint
            is_best = currscore > best_rsum
            best_rsum = max(currscore, best_rsum)
            if is_best:
                torch.save(model.state_dict(), ckpt_path)

            all_transformed_text_embs = []
            all_transformed_mel_embs = []
            for val_text_embs, val_spkr_embs in val_retrieval_loader:
                val_text_embs = val_text_embs.cuda(args.cuda)
                val_spkr_embs = val_spkr_embs.cuda(args.cuda)
                o1, o2 = model.forward(val_text_embs, val_spkr_embs)
                loss = loss_fn(o1, o2)
                val_losses.append(loss.item())
                all_transformed_text_embs.append(o1.cpu().numpy())
                all_transformed_mel_embs.append(o2.cpu().numpy())
            all_transformed_text_embs = np.concatenate(all_transformed_text_embs, axis=0)
            all_transformed_mel_embs = np.concatenate(all_transformed_mel_embs, axis=0)

            (r1i, r5i, r10i, r50, r100, medri) = i2t(all_transformed_mel_embs, all_transformed_text_embs)
            logging.info("Test: %.1f%%, %.1f%%, %.1f%%, %.1f%%, %.1f%%, %d" %
                        (r1i, r5i, r10i, r50, r100, int(medri)))
            
            print('%d loss: train %.4f, val %.4f' % (epoch, np.mean(train_losses), np.mean(val_losses)))


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
