'''
Peter Wu
peterw1@andrew.cmu.edu
'''

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import numpy as np

from collections import Counter 


def i2t(mel_embs, text_embs, npts=None):
    """Spectrogram->Text Retrieval

    R@K is Recall@K (high is good). For median rank, low is good.
    Details described in https://arxiv.org/pdf/1411.2539.pdf.

    Args:
        mel_embs: spectrogram embeddings, np array with shape (num_data, emb_dim)
            where num_data is either number of dev or test datapoints
        text_embs: text embeddings, np array with shape shape (num_data, emb_dim)

    Return:
        r1: R@1
        r5: R@5
        r10: R@10
        medr: median rank
    """
    ranks = np.zeros(len(mel_embs))
    mnorms = np.sqrt(np.sum(mel_embs**2,axis=1)[None]).T
    mel_embs = mel_embs / mnorms
    tnorms = np.sqrt(np.sum(text_embs**2,axis=1)[None]).T
    text_embs = text_embs / tnorms
    
    for index in range(len(mel_embs)):
        im = mel_embs[index].reshape(1, mel_embs.shape[1]) # shape (1, 1024)
        d = np.dot(im, text_embs.T).flatten()
        inds = np.argsort(d)[::-1]
        rank = 1e20
        tmp = np.where(inds == index)[0][0]
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return (r1, r5, r10, r50, r100, medr)


def t2i(mel_embs, text_embs):
    """Text->Spectrogram Retrieval

    R@K is Recall@K (high is good). For median rank, low is good.
    Details described in https://arxiv.org/pdf/1411.2539.pdf.

    Args:
        mel_embs: spectrogram embeddings, np array with shape (num_data, emb_dim)
            where num_data is either number of dev or test datapoints
        text_embs: text embeddings, np array with shape shape (num_data, emb_dim)

    Return:
        r1: R@1
        r5: R@5
        r10: R@10
        medr: median rank
    """
    ranks = np.zeros(len(mel_embs))
    mnorms = np.sqrt(np.sum(mel_embs**2,axis=1)[None]).T
    mel_embs = mel_embs / mnorms
    tnorms = np.sqrt(np.sum(text_embs**2,axis=1)[None]).T
    text_embs = text_embs / tnorms

    for index in range(len(mel_embs)):
        queries = text_embs[index:index+1, :]
        d = np.dot(queries, mel_embs.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[index] = np.where(inds[i] == index)[0][0]
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return (r1, r5, r10, r50, r100, medr)

all_categories = np.load('categories.npy')
category_to_i = {c:i for i,c in enumerate(all_categories)}


def top_missed_categories(mel_embs, text_embs, categories):
    """Top missed categories

    Args:
        mel_embs: spectrogram embeddings, np array with shape (num_data, emb_dim)
            where num_data is either number of dev or test datapoints
        text_embs: text embeddings, np array with shape shape (num_data, emb_dim)
        categories: list of strings, categories[i] is category of text_embs[i]

    Return:
        - number of missed categories
        - list of 3 most missed categories wrt r@100 threshold (list of strings)
        - list of 3 most mistaken pairs (list of (string,string) pairs)
        - confusion matrix
    """
    ranks = np.zeros(len(mel_embs)) # num_samples
    mnorms = np.sqrt(np.sum(mel_embs**2,axis=1)[None]).T
    mel_embs = mel_embs / mnorms # num_samples, emb_dim
    tnorms = np.sqrt(np.sum(text_embs**2,axis=1)[None]).T
    text_embs = text_embs / tnorms # num_samples, emb_dim
    mistaken_idxs = np.zeros(len(mel_embs), dtype=int)

    for index in range(len(mel_embs)):
        queries = text_embs[index:index+1, :] # 1, emb_dim
        d = np.dot(queries, mel_embs.T) # 1, num_samples
        inds = np.zeros(d.shape) # 1, num_samples
        inds[0] = np.argsort(d[0])[::-1] # num_samples
        mistaken_idxs[index] = int(inds[0][0])
        ranks[index] = np.where(inds[0] == index)[0][0] # int

    missed_idxs = 1 - np.where(ranks < 100)[0]
    missed_counts = {c: 0 for c in all_categories}
    for i,c in zip(missed_idxs, categories):
        if i == 1:
            missed_counts[c] += 1
    k = Counter(missed_counts)
    high = k.most_common(3)

    num_missed_categories = 0
    conf_mat = np.zeros((len(all_categories), len(all_categories)))
    mistaken_mat = np.zeros((len(all_categories), len(all_categories)))
    for i1, i2 in enumerate(mistaken_idxs):
        c1 = categories[i1]
        c2 = categories[i2]
        c1i = category_to_i[c1]
        c2i = category_to_i[c2]
        conf_mat[c1i][c2i] += 1
        if c1i != c2i:
            num_missed_categories += 1
        if c1i < c2i:
            mistaken_mat[c1i][c2i] += 1
        elif c1i > c2i:
            mistaken_mat[c2i][c1i] += 1
    most_mistaken_cidxs = list(np.unravel_index(np.argsort(mistaken_mat.ravel())[-3:], mistaken_mat.shape))
    most_mistaken_pairs = []
    for p1, p2 in zip(*most_mistaken_cidxs):
        most_mistaken_pairs.append((all_categories[p1], all_categories[p2]))

    return num_missed_categories, [h[0] for h in high], most_mistaken_pairs, conf_mat


def top_mistaken(mel_embs, text_embs, categories):
    """Top missed categories

    Args:
        mel_embs: spectrogram embeddings, np array with shape (num_data, emb_dim)
            where num_data is either number of dev or test datapoints
        text_embs: text embeddings, np array with shape shape (num_data, emb_dim)
        categories: list of strings, categories[i] is category of text_embs[i]

    Return:
        - number of missed categories
        - list of 3 most missed categories wrt r@100 threshold (list of strings)
        - list of 3 most mistaken pairs (list of (string,string) pairs)
        - confusion matrix
    """
    ranks = np.zeros(len(mel_embs)) # num_samples
    mnorms = np.sqrt(np.sum(mel_embs**2,axis=1)[None]).T
    mel_embs = mel_embs / mnorms # num_samples, emb_dim
    tnorms = np.sqrt(np.sum(text_embs**2,axis=1)[None]).T
    text_embs = text_embs / tnorms # num_samples, emb_dim
    mistaken_idxs = np.zeros(len(mel_embs), dtype=int)

    for index in range(len(mel_embs)):
        queries = text_embs[index:index+1, :] # 1, emb_dim
        d = np.dot(queries, mel_embs.T) # 1, num_samples
        inds = np.zeros(d.shape) # 1, num_samples
        inds[0] = np.argsort(d[0])[::-1] # num_samples
        mistaken_idxs[index] = int(inds[0][0])
        ranks[index] = np.where(inds[0] == index)[0][0] # int

    missed_idxs = 1 - np.where(ranks < 100)[0]
    missed_counts = {c: 0 for c in all_categories}
    for i,c in zip(missed_idxs, categories):
        if i == 1:
            missed_counts[c] += 1
    k = Counter(missed_counts)

    num_missed_categories = 0
    conf_mat = np.zeros((len(all_categories), len(all_categories)))
    mistaken_mat = np.zeros((len(all_categories), len(all_categories)))
    for i1, i2 in enumerate(mistaken_idxs):
        c1 = categories[i1]
        c2 = categories[i2]
        c1i = category_to_i[c1]
        c2i = category_to_i[c2]
        conf_mat[c1i][c2i] += 1
        if c1i != c2i:
            num_missed_categories += 1
        if c1i < c2i:
            mistaken_mat[c1i][c2i] += 1
        elif c1i > c2i:
            mistaken_mat[c2i][c1i] += 1
    most_mistaken_cidxs = list(np.unravel_index(np.argsort(mistaken_mat.ravel())[-30:], mistaken_mat.shape))
    most_mistaken_pairs = []
    most_mistaken_counts = []
    for p1, p2 in zip(*most_mistaken_cidxs):
        most_mistaken_counts.append(mistaken_mat[p1][p2])
        most_mistaken_pairs.append((all_categories[p1], all_categories[p2]))

    return num_missed_categories, most_mistaken_counts, most_mistaken_pairs


def save_conf_mat(conf_mat, conf_mat_path='conf_mat.png'):
    '''
    Saves confusion matrix at conf_mat_path
    '''
    plt.figure(figsize=(20, 20))
    im = plt.imshow(conf_mat)
    plt.colorbar(im, fraction=0.02, pad=0.01)
    plt.savefig(conf_mat_path)
    plt.close()


def window_mel(mel, window_size=512):
    '''
    Args:
        mel: np array with shape (seq_len, num_mels), where num_mels = 80
        window_size: int
    
    Return:
        new_mel: np array with shape (seq_len, window_size)
    '''
    num_frames = mel.shape[0]
    # num_mels = 80
    num_mels = mel.shape[1]
    if num_frames > window_size:
        start_i = np.random.randint(0, high=num_frames-window_size+1)
        mel = mel[start_i:start_i+window_size, :]
    new_mel = np.concatenate([np.zeros([1,num_mels], np.float32), mel[:-1,:]], axis=0)
    return new_mel


def mk_idx_mat(embs, num_neighbors=10):
    '''
    Args:
        embs: np array with shape (batch_size, emb_dim)
            contains text embeddings of each datapoint
    
    Return:
        idxs: np array with shape (batch_size, num_neighbors)
    '''
    ranks = np.zeros(len(embs))
    enorms = np.sqrt(np.sum(embs**2,axis=1)[None]).T
    embs = embs / enorms
    idxs = []
    for index in range(len(embs)):
        queries = embs[index:index+1, :] # 1, emb_dim
        d = np.dot(queries, embs.T) # 1, num_samples
        inds = np.zeros(d.shape) # 1, num_samples
        inds[0] = np.argsort(d[0])[::-1] # num_samples
        inds = inds[0][:num_neighbors]
        idxs.append(inds)
    idxs = np.stack(idxs)
    return idxs