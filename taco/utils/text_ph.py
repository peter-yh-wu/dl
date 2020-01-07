'''
Based on https://github.com/CorentinJ/Real-Time-Voice-Cloning

Peter Wu
peterw1@andrew.cmu.edu
'''

import numpy as np
import re


# Mappings from symbol to numeric ID and vice versa:
_pad = "_"
_eos = "~"
_unk = "#"
ph_vocab = np.load('ph_vocab.npy')
ph_vocab = [_pad, _eos, _unk] + list(ph_vocab)
_symbol_to_id = {s: i for i, s in enumerate(ph_vocab)}
_id_to_symbol = {i: s for i, s in enumerate(ph_vocab)}

space_token = len(ph_vocab)
symbols = list(ph_vocab) + [' ']


def symbol_to_id(ph):
    if ph in _symbol_to_id:
        return _symbol_to_id[ph]
    return _symbol_to_id[_unk]


def text_to_sequence(text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    Called by feeder_ph.py and tacotron2_ph.py

    Args:
        text: string to convert to a sequence (as formated in last part of each train_ph.txt line)
            sequence of space-separated phonemes, with commas separating words
            (no spaces on either side of commas) 

    Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []
    ph_list_by_word = text.split(',')
    for ph_word in ph_list_by_word:
        phs = ph_word.split()
        for ph in phs:
            sequence.append(symbol_to_id(ph))
        sequence.append(space_token)
    sequence[-1] = 1 # replace last space token with EOS token
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string

    Called by train_ph.py, just used for logging purposes

    Args:
        sequence: iterable of ints

    Return:
        string, namely text with same format as original phoneme sequence in train_ph.txt
    '''
    result = ""
    for symbol_id in sequence:
        if symbol_id == space_token:
            result += ','
        else: # elif symbol_id != _eos:
            s = _id_to_symbol[symbol_id]
            if len(result) > 0 and result[-1] != ',':
                result += ' '
            result += s
    return result