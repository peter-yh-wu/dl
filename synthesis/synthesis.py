'''
Based on https://github.com/soobinseo/Transformer-TTS

Peter Wu
peterw1@andrew.cmu.edu
'''

import numpy as np
import os
import torch
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse

from utils import spectrogram2wav


def load_checkpoint(step, model_name="transformer"):
    state_dict = torch.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict


def synthesis(text, args):
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(args.step1, "transformer"))
    m_post.load_state_dict(load_checkpoint(args.step2, "postnet"))

    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    
    mel_input = torch.zeros([1,1, 80]).cuda()

    pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()

    m=m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)
    
    pbar = tqdm(range(args.max_len))
    with torch.no_grad():
        for i in pbar:
            pos_mel = torch.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            mel_input = torch.cat([mel_input, postnet_pred[:,-1:,:]], dim=1)

        mag_pred = m_post.forward(postnet_pred)
        
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    if not os.path.exists(hp.sample_path):
        os.makedirs(hp.sample_path)
    write(hp.sample_path + "/test.wav", hp.sr, wav)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--step1', type=int, help='Global step to restore checkpoint', default=4000) # was 172000
    parser.add_argument('--step2', type=int, help='Global step to restore checkpoint', default=500) # was 100000
    parser.add_argument('--prompt', type=str, help='', default='walking a dog in the park')
    parser.add_argument('--max_len', type=int, help='', default=1024) # was 400

    args = parser.parse_args()
    synthesis(args.prompt,args)
