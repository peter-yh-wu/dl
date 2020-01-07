'''
Based on https://github.com/soobinseo/Transformer-TTS

Peter Wu
peterw1@andrew.cmu.edu
'''

import csv
import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal
import torch
import math


class AudioDatasets(Dataset):

    def __init__(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(parent_dir, 'data')
        self.mel_dir = os.path.join(data_dir, 'mel')
        mel_files = os.listdir(self.mel_dir)
        self.mel_files = [f for f in mel_files if f.endswith('.pt.npy')]

        ids = [] # index 0
        captions = [] # index 5
        csv_path = os.path.join(data_dir, 'train.csv')
        with open(csv_path, 'r') as csvfile:
            raw_csv = csv.reader(csvfile)
            for row in raw_csv:
                ids.append(row[0])
                captions.append(row[5])

        id_to_caption = {ci: cap for ci, cap in zip(ids, captions)}

        self.texts = []
        new_mel_files = []
        for f in self.mel_files:
            ci = f[:-7]
            if ci in id_to_caption:
                self.texts.append(id_to_caption[ci])
                new_mel_files.append(f)
        self.mel_files = new_mel_files

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel_file = self.mel_files[idx]
        mel_path = os.path.join(self.mel_dir, mel_file)
        
        text = self.texts[idx]

        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        mel = np.load(mel_path)
        num_frames = mel.shape[0]
        window_size = int(hp.seq_len/2)
        if num_frames > window_size:
            start_i = np.random.randint(0, high=num_frames-window_size+1)
            mel = mel[start_i:start_i+window_size, :]
        mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text}

        return sample
    

class PostDatasets(Dataset):

    def __init__(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(parent_dir, 'data')
        self.mel_dir = os.path.join(data_dir, 'mel')
        mel_files = os.listdir(self.mel_dir)
        self.mel_files = [f for f in mel_files if f.endswith('.pt.npy')]

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel_file = self.mel_files[idx]
        mel_path = os.path.join(self.mel_dir, mel_file)
        mag_path = os.path.join(self.mel_dir, mel_file[:-7] + '.mag.npy')

        mel = np.load(mel_path)
        mag = np.load(mag_path)
        sample = {'mel':mel, 'mag':mag}

        return sample
    
def collate_fn_transformer(batch):
    if isinstance(batch[0], collections.Mapping):

        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text= [d['pos_text'] for d in batch]
        
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)

        text = _prepare_data(text).astype(np.int32)
        mel = _pad_mel(mel)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)


        return torch.LongTensor(text), torch.FloatTensor(mel), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))
    
def collate_fn_postnet(batch):
    if isinstance(batch[0], collections.Mapping):

        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]
        
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return torch.FloatTensor(mel), torch.FloatTensor(mag)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)


def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params


def get_dataset():
    return AudioDatasets()


def get_post_dataset():
    return PostDatasets()


def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

