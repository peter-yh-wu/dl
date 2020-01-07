'''
Peter Wu
peterw1@andrew.cmu.edu
'''

import numpy as np
import os
import pickle
import random
import torch

from torch.utils.data import Dataset, DataLoader


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class TextSpeakerEmbDataset(Dataset):
    def __init__(self, phase, retrieval=False, window_size=512, audio='mel'):
        self.window_size = window_size

        text_file_path = 'text/%s.txt' % phase
        with open(text_file_path, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_list = [l.split('\t') for l in lines]
        ids = [l[0] for l in l_list]
        texts = [l[1] for l in l_list]
        
        category_file_path = 'text/%s_categories.txt' % phase
        with open(category_file_path, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_list = [l.split('\t') for l in lines]
        self.categories = [l[1] for l in l_list]

        self.audio = audio
        self.text_emb_paths = ['text_embs/'+curr_id+'.npy' for curr_id in ids]
        if audio == 'mel':
            self.mel_emb_paths = ['mel/'+curr_id+'.pt.npy' for curr_id in ids]
        else:
            self.mel_emb_paths = ['cqt/'+curr_id+'.npy' for curr_id in ids]

    def __len__(self):
        return len(self.mel_emb_paths)
    
    def __getitem__(self, index):
        mel_emb_path = self.mel_emb_paths[index]
        frames = np.load(mel_emb_path)
        if self.audio == 'cqt':
            frames = np.transpose(frames, [1,0]).real.astype(np.float32)
        num_frames = frames.shape[0]
        if num_frames > self.window_size:
            start_i = np.random.randint(0, high=num_frames-self.window_size+1)
            frames = frames[start_i:start_i+self.window_size, :]
        else:
            frames = np.pad(frames, ((0,self.window_size-num_frames), (0,0)), mode='constant', constant_values=0)
        text_emb_path = self.text_emb_paths[index]
        text_emb = np.load(text_emb_path)[0]
        return text_emb, frames, self.categories[index]


def mk_dataloader(phase, retrieval=False, batch_size=64, shuffle=True, num_workers=4, audio='mel'):  
    window_size = 512
    if audio == 'cqt':
        window_size = 431
    dataset = TextSpeakerEmbDataset(phase, retrieval=retrieval, window_size=window_size, audio=audio)
    dataloader = DataLoader(dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers)
    return dataloader


class ConversionDataset(Dataset):
    def __init__(self, phase, window_size=512, audio='mel'):
        self.window_size = window_size

        text_file_path = 'text/train.txt'
        with open(text_file_path, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_list = [l.split('\t') for l in lines]
        ids = [l[0] for l in l_list]
        texts = [l[1] for l in l_list]
        
        category_file_path = 'text/train_categories.txt'
        with open(category_file_path, 'r') as inf:
            lines = inf.readlines()
        lines = [l.strip() for l in lines]
        l_list = [l.split('\t') for l in lines]
        self.categories = [l[1] for l in l_list]

        self.audio = audio
        self.text_emb_paths = ['text_embs/'+curr_id+'.npy' for curr_id in ids]
        if audio == 'mel':
            self.mel_emb_paths = ['mel/'+curr_id+'.pt.npy' for curr_id in ids]
        else:
            self.mel_emb_paths = ['cqt/'+curr_id+'.npy' for curr_id in ids]
    
        idx_mat = np.load('idx_mat.npy').astype(int)
        idx_mat = [mat for mat in idx_mat]
        random.Random(0).shuffle(idx_mat)
        if phase == 'train':
            idx_mat = idx_mat[:7000]
        elif phase == 'dev':
            idx_mat = idx_mat[7000:7500]
        else:
            idx_mat = idx_mat[7500:]
        self.idx_tuples = []
        for i, mat in enumerate(idx_mat):
            for i2 in mat:
                if phase == 'train':
                    tup = (i, i2)
                elif phase == 'dev':
                    tup = (i+7000, i2)
                else:
                    tup = (i+7500, i2)
                self.idx_tuples.append(tup)

    def __len__(self):
        return len(self.idx_tuples)
    
    def __getitem__(self, idx):
        tup = self.idx_tuples[idx]
        index, index2 = tup
        mel_emb_path = self.mel_emb_paths[index]
        frames = np.load(mel_emb_path)
        if self.audio == 'cqt':
            frames = np.transpose(frames, [1,0]).real.astype(np.float32)
        num_frames = frames.shape[0]
        if num_frames > self.window_size:
            start_i = np.random.randint(0, high=num_frames-self.window_size+1)
            frames = frames[start_i:start_i+self.window_size, :]
        else:
            frames = np.pad(frames, ((0,self.window_size-num_frames), (0,0)), mode='constant', constant_values=0)
        mel_emb_path = self.mel_emb_paths[index2]
        frames2 = np.load(mel_emb_path)
        if self.audio == 'cqt':
            frames2 = np.transpose(frames2, [1,0]).real.astype(np.float32)
        num_frames = frames2.shape[0]
        if num_frames > self.window_size:
            start_i = np.random.randint(0, high=num_frames-self.window_size+1)
            frames2 = frames2[start_i:start_i+self.window_size, :]
        else:
            frames2 = np.pad(frames2, ((0,self.window_size-num_frames), (0,0)), mode='constant', constant_values=0)
        return frames, frames2


def mk_convert_dataloader(phase, batch_size=64, shuffle=True, num_workers=4, audio='mel'):  
    window_size = 512
    if audio == 'cqt':
        window_size = 431
    dataset = ConversionDataset(phase, window_size=window_size, audio=audio)
    dataloader = DataLoader(dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers)
    return dataloader
