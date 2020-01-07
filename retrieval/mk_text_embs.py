'''
Peter Wu
peterw1@andrew.cmu.edu
'''

import numpy as np
import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def mk_embs(texts, out_paths):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    model = model.cuda()

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    for text, out_path in zip(texts, out_paths):
        # Load pre-trained model tokenizer (vocabulary)

        marked_text = "[CLS] " + text + " [SEP]" # text is a stri

        tokenized_text = tokenizer.tokenize(marked_text)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        segments_ids = [1] * len(tokenized_text)
        segments_tensors = torch.tensor([segments_ids]).cuda()

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

        sentence_embedding = torch.mean(encoded_layers[11], 1) # 768 length vector
        np.save(out_path, sentence_embedding.cpu().numpy())

emb_dir = 'text_embs'
if not os.path.exists(emb_dir):
    os.makedirs(emb_dir)

texts = []
out_paths = []
text_file_paths = ['text/train.txt', 'text/dev.txt', 'text/test.txt']
for text_file_path in text_file_paths:
    with open(text_file_path, 'r') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    l_list = [l.split('\t') for l in lines]
    curr_ids = [l[0] for l in l_list]
    curr_texts = [l[1] for l in l_list]
    curr_out_paths = [os.path.join(emb_dir, curr_id+'.npy') for curr_id in curr_ids]
    texts += curr_texts
    out_paths += curr_out_paths

mk_embs(texts, out_paths)