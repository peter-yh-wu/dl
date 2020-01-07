'''
Based on https://github.com/gabrielhuang/reptile-pytorch

Peter Wu
peterw1@andrew.cmu.edu
'''

import argparse
import json
import math
import numpy as np
import os
import random
import torch
import tqdm

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from models import OmniglotModel
from objectives import CCALoss
from data_utils import *
from utils import svm_classify


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def Variable_(tensor, *args_, **kwargs):
    '''
    Make variable cuda depending on the arguments
    '''
    # Unroll list or tuple
    if type(tensor) in (list, tuple):
        return [Variable_(t, *args_, **kwargs) for t in tensor]
    # Unroll dictionary
    if isinstance(tensor, dict):
        return {key: Variable_(v, *args_, **kwargs) for key, v in tensor.items()}
    # Normal tensor
    variable = tensor # Variable(tensor, *args_, **kwargs)
    variable = variable.cuda(args.cuda)
    return variable

# Parsing
parser = argparse.ArgumentParser('Train reptile on omniglot')

# Mode

# - Training params
parser.add_argument('--classes', default=5, type=int, help='classes in base-task (N-way)')
parser.add_argument('--shots', default=5, type=int, help='shots per class (K-shot)')
parser.add_argument('--train-shots', default=10, type=int, help='train shots')
parser.add_argument('--meta-iterations', default=10000, type=int, help='number of meta iterations')
parser.add_argument('--start-meta-iteration', default=0, type=int, help='start iteration')
parser.add_argument('--iterations', default=5, type=int, help='number of base iterations')
parser.add_argument('--cca-iters', default=5, type=int, help='number of base iterations')
parser.add_argument('--test-iterations', default=50, type=int, help='number of base iterations')
parser.add_argument('--batch', default=4, type=int, help='minibatch size in base task') # 10
parser.add_argument('--meta-lr', default=1., type=float, help='meta learning rate')
parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')

# - General params
parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
parser.add_argument('--validate-every', default=100, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--cuda', default=1, type=int, help='cuda device')

# Do some processing
args = parser.parse_args()
print(args)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Load data
# Resize is done by the MetaDataset because the result can be easily cached
folder = MetaOmniglotFolder(size=(360, 240), cache=ImageCache(),
                              transform_image=transform_image,
                              transform_label=transform_label)
meta_train, meta_val, meta_test = split_omniglot(folder, args.validation)

print('Meta-Train characters', len(meta_train))
print('Meta-Val characters', len(meta_val))
print('Meta-Test characters', len(meta_test))

# Loss
cross_entropy = nn.CrossEntropyLoss() # nn.NLLLoss()
def get_loss(prediction, labels):
    return cross_entropy(prediction, labels)


def do_learning(net, optimizer, train_iter, iterations):
    net.train()
    for iteration in range(iterations):
        # Sample minibatch
        imgs, texts, labels = Variable_(next(train_iter))

        # Forward pass
        prediction = net(texts)

        # Get loss
        loss = get_loss(prediction, labels)

        # Backward pass - Update fast net
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item() # .data[0]


def do_evaluation(net, test_iter, iterations):
    losses = []
    accuracies = []
    net.eval()
    for iteration in range(iterations):
        # Sample minibatch
        imgs, texts, labels = Variable_(next(test_iter))

        # Forward pass
        prediction = net(texts)

        # Get loss
        loss = get_loss(prediction, labels)

        # Get accuracy
        argmax = net.predict(prediction)
        accuracy = (argmax == labels).float().mean()

        losses.append(loss.item()) # .data[0])
        accuracies.append(accuracy.item()) # .data[0])

    return np.mean(losses), np.mean(accuracies)


def get_optimizer(net, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def get_cca_optimizer(net, state=None):
    optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3, weight_decay=1e-5)
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_cca(net, cca_loss, optimizer, train_iter, iterations):
    net.train()
    train_losses = []
    for iteration in range(iterations):
        optimizer.zero_grad()
        imgs, texts, labels = Variable_(next(train_iter))
        # with autograd.detect_anomaly():
        o1, o2 = net.forward_cca(imgs, texts)
        loss = cca_loss.loss(o1, o2)
        train_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)
        optimizer.step()


def test_cca(net, cca_loss, optimizer, train_iter, val_iter, train_iterations, val_iterations):
    '''
    Return:
        test_loss: number
        acc: number
    '''
    train_losses = []
    train_data = []
    train_label = []
    net.train()
    for iteration in range(train_iterations):
        optimizer.zero_grad()
        imgs, texts, labels = Variable_(next(train_iter))
        o1, o2 = net.forward_cca(imgs, texts)
        loss = cca_loss.loss(o1, o2)
        train_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)
        optimizer.step()
        train_data.append(o1) # o1.clone().detach().cpu().numpy())
        train_label.append(labels) # labels.clone().detach().cpu().numpy())
    train_loss = np.mean(train_losses)

    losses = []
    val_data = []
    val_label = []
    with torch.no_grad():
        net.eval()
        for iteration in range(val_iterations):
            imgs, texts, labels = Variable_(next(val_iter))
            o1, o2 = net.forward_cca(imgs, texts)
            val_data.append(o1) # o1.detach().cpu().numpy())
            val_label.append(labels) # labels.detach().cpu().numpy())
            loss = cca_loss.loss(o1, o2)
            losses.append(loss.item())
        train_data = torch.cat(train_data, 0)
        val_data = torch.cat(val_data, 0)
        train_label = torch.cat(train_label, 0)
        val_label = torch.cat(val_label, 0)
        train_data = train_data.cpu().numpy()
        val_data = val_data.cpu().numpy()
        train_label = train_label.cpu().numpy()
        val_label = val_label.cpu().numpy()
    test_loss = np.mean(losses)
    acc = svm_classify(train_data, val_data, train_label, val_label)
    
    return train_loss, test_loss, acc

fc_dim = 32

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(parent_dir, 'data')
word2index_path = os.path.join(data_dir, 'word2index.pkl')
emb_mat_path = os.path.join(data_dir, 'emb_mat.npy')
word2index = load_pkl(word2index_path)
emb_mat = np.load(emb_mat_path)
vocab_size = len(word2index)+2

# Build model, optimizer, and set states
meta_net = OmniglotModel(fc_dim, args.classes, vocab_size, emb_mat, args.cuda)
meta_net.cuda(args.cuda)
meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=args.meta_lr)
state = None
cca_state = None

# fast_optimizer
cca_optimizer = torch.optim.RMSprop(meta_net.parameters(), lr=1e-3, weight_decay=1e-5)

outdim_size = fc_dim # 32
use_all_singular_values = False
loss_fn = nn.CrossEntropyLoss()
cca_loss = CCALoss(outdim_size, use_all_singular_values, args.cuda)

# For Evaluation Datasets
idx_dict_path = os.path.join(data_dir, 'idx_dict.npy')
idx_dict = np.load(idx_dict_path, allow_pickle=True)
idx_dict = idx_dict[()]

for meta_iteration in range(args.start_meta_iteration, args.meta_iterations):
    # Update learning rate
    meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iterations))
    set_learning_rate(meta_optimizer, meta_lr)

    # Clone model
    net = meta_net.clone()
    optimizer = get_optimizer(net, state)

    # Sample base task from Meta-Train
    train = meta_train.get_random_task(args.classes, args.train_shots or args.shots)
    train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True, collate_fn=collate_img_text))

    # Update fast net
    loss = do_learning(net, optimizer, train_iter, args.iterations)
    state = optimizer.state_dict()  # save optimizer state

    # Update slow net
    meta_net.point_grad_to(net)
    meta_optimizer.step()

    str_output = ''

    # Meta-Evaluation
    if meta_iteration % args.validate_every == 0:
        metrics = []
        for (meta_dataset, mode) in [(meta_train, 'train'), (meta_val, 'val'), (meta_test, 'test')]:
            mode = 'meta_'+mode
            curr_idx_dict = idx_dict[mode]
            meta_losses = []
            meta_accuracies = []
            for task_idx_dict in curr_idx_dict:
                character_indices = task_idx_dict['character_indices']
                all_curr_idxs = task_idx_dict['all_curr_idxs']
                new_train_idxs = task_idx_dict['new_train_idxs']
                new_test_idxs = task_idx_dict['new_test_idxs']
                train, val = meta_dataset.get_task_split(character_indices, all_curr_idxs, new_train_idxs, new_test_idxs, train_K=5)

                train_iter = make_infinite(DataLoader(train, args.batch, shuffle=False, collate_fn=collate_img_text))
                val_iter = make_infinite(DataLoader(val, args.batch, shuffle=False, collate_fn=collate_img_text))

                # Base-train (tuning)
                net = meta_net.clone()
                optimizer = get_optimizer(net, state)  # do not save state of optimizer
                meta_train_loss = do_learning(net, optimizer, train_iter, args.test_iterations)

                # Base-test: compute meta-loss, which is base-validation error
                num_iter = int(math.ceil(len(val)/args.batch))
                meta_loss, meta_accuracy = do_evaluation(net, val_iter, num_iter)
                meta_losses.append(meta_loss)
                meta_accuracies.append(meta_accuracy)

            # (Logging)
            metrics.append(meta_train_loss)
            metrics.append(np.mean(meta_losses))
            metrics.append(np.mean(meta_accuracies))
        str_output += ' '.join(['%.4f' % f for f in metrics])

    str_output += ' '

    # --- meta train cca ---

    # Clone model
    net = meta_net.clone()
    optimizer = get_cca_optimizer(net, cca_state)

    # Sample base task from Meta-Train
    train = meta_train.get_random_task(args.classes, args.train_shots or args.shots)
    train_iter = make_infinite(DataLoader(train, args.batch, shuffle=True, collate_fn=collate_img_text))

    train_cca(net, cca_loss, optimizer, train_iter, args.cca_iters)
    cca_state = optimizer.state_dict()

    # --- meta test cca ---

    if meta_iteration % args.validate_every == 0:
        metrics = []
        for (meta_dataset, mode) in [(meta_train, 'train'), (meta_val, 'val'), (meta_test, 'test')]:
            mode = 'meta_'+mode
            curr_idx_dict = idx_dict[mode]
            meta_train_losses = []
            meta_losses = []
            meta_accuracies = []
            for task_idx_dict in curr_idx_dict:
                character_indices = task_idx_dict['character_indices']
                all_curr_idxs = task_idx_dict['all_curr_idxs']
                new_train_idxs = task_idx_dict['new_train_idxs']
                new_test_idxs = task_idx_dict['new_test_idxs']
                train, val = meta_dataset.get_task_split(character_indices, all_curr_idxs, new_train_idxs, new_test_idxs, train_K=5)
                
                train_iter = make_infinite(DataLoader(train, args.batch, shuffle=False, collate_fn=collate_img_text))
                val_iter = make_infinite(DataLoader(val, args.batch, shuffle=False, collate_fn=collate_img_text))

                # Base-train (tuning)
                net = meta_net.clone()
                num_iter = int(math.ceil(len(val)/args.batch))
                optimizer = get_cca_optimizer(net, cca_state)
                train_loss, test_loss, test_acc = test_cca(net, cca_loss, optimizer, train_iter, val_iter, args.test_iterations, num_iter)
                meta_train_losses.append(train_loss)
                meta_losses.append(test_loss)
                meta_accuracies.append(test_acc)

            # (Logging)
            metrics.append(np.mean(meta_train_losses))
            metrics.append(np.mean(meta_losses))
            metrics.append(np.mean(meta_accuracies))

        str_output += ' '.join(['%.4f' % f for f in metrics])
        print(str_output)