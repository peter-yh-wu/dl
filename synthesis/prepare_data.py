import numpy as np
import pandas as pd
import os
import librosa

from functools import partial
from multiprocessing.pool import Pool

import hyperparams as hp

from utils import get_spectrograms


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(parent_dir, 'data')
wav_dir = os.path.join(data_dir, 'wav')
wav_files = os.listdir(wav_dir)
wav_files = [f for f in wav_files if f.endswith('.wav')]
out_dir = os.path.join(data_dir, 'mel')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def load_wav(filename):
    return librosa.load(filename, sr=hp.sample_rate)


def process_file(wav_file):
    print(wav_file)
    wav_path = os.path.join(wav_dir, wav_file)
    mel, mag = get_spectrograms(wav_path)
    mel_path = os.path.join(out_dir, wav_file[:-4] + '.pt')   
    mag_path = os.path.join(out_dir, wav_file[:-4] + '.mag')    
    np.save(mel_path, mel)
    np.save(mag_path, mag)


pool = Pool(72)
metadata = pool.map(process_file, wav_files)
pool.close()
pool.join()