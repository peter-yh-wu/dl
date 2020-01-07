import numpy as np
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
    
    mel_input = np.load('3_0.pt.npy')
    
    pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()

    m=m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)
    
    with torch.no_grad():
        mag_pred = m_post.forward(torch.from_numpy(mel_input).unsqueeze(0).cuda())
        
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/test.wav", hp.sr, wav)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step1', type=int, help='Global step to restore checkpoint', default=4000) # was 172000
    parser.add_argument('--step2', type=int, help='Global step to restore checkpoint', default=500) # was 100000
    parser.add_argument('--max_len', type=int, help='', default=1024) # was 400

    args = parser.parse_args()
    synthesis("walking a dog in the park",args)
