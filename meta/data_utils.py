'''
Based on https://github.com/gabrielhuang/reptile-pytorch

Peter Wu
peterw1@andrew.cmu.edu
'''

import os
import numpy as np
import torch

from nltk.tokenize import word_tokenize
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import load_pkl, mk_label_map, transform_label


def read_image(path, size=None):
    img = Image.open(path, mode='r')
    if size is not None:
        img = img.resize(size)
    img = np.transpose(img, [2, 0, 1])
    img = img/255.
    return img


def collate_img_text(batch):
    '''
    Args:
        batch: list of (text, label) pairs
    '''
    imgs = []
    texts = []
    labels = []
    max_len = 0
    for (_, text, _) in batch:
        x_len = len(text)
        if x_len > max_len:
            max_len = x_len
    for (img, text, label) in batch:
        imgs.append(img)
        new_x = np.pad(text, (0, max_len-len(text)), 'constant', constant_values=0)
        texts.append(torch.tensor(new_x))
        labels.append(label)
    imgs = torch.stack(imgs)
    texts = torch.stack(texts)
    return imgs, texts.long(), torch.tensor(labels)


def tokenize_text(text, word2index):
    '''
    Args:
        text: string
    '''
    text_list = word_tokenize(text)

    new_text_list = []
    for word in text_list:
        if word in word2index:
            new_text_list.append(word2index[word])
        else:
            new_text_list.append(1)
    text_list = new_text_list
    return text_list


# Default transforms
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(parent_dir, 'data')
word2index_path = os.path.join(data_dir, 'word2index.pkl')
word2index = load_pkl(word2index_path)
transform_text = transforms.Compose([lambda x: tokenize_text(x, word2index)])


class ImageCache(object):
    def __init__(self):
        self.cache = {}

    def read_image(self, path, size=None):
        key = (path, size)
        if key not in self.cache:
            self.cache[key] = read_image(path, size)
        else:
            pass  #print 'reusing cache', key
        return self.cache[key]


class FewShot(Dataset):
    '''
    Dataset for K-shot N-way classification
    '''
    def __init__(self, paths, texts, meta=None, parent=None):
        self.paths = paths
        self.texts = texts
        self.meta = {} if meta is None else meta
        self.parent = parent

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]['path']
        if self.parent.cache is None:
            image = read_image(path, self.parent.size)
        else:
            image = self.parent.cache.read_image(path, self.parent.size)
        if self.parent.transform_image is not None:
            image = self.parent.transform_image(image)
        
        text = self.texts[idx]['text']
        if self.parent.transform_text is not None:
            text = self.parent.transform_text(text)
        
        label = self.paths[idx]
        if self.parent.transform_label is not None:
            label = self.parent.transform_label(label)
        
        return image, text, label


class AbstractMetaOmniglot(object):

    def __init__(self, characters_list, texts_list, cache=None, size=(28, 28),
                 transform_image=None, transform_label=None):
        self.characters_list = characters_list
        self.cache = cache
        self.size = size
        self.transform_image = transform_image
        self.texts_list = texts_list
        self.transform_text = transform_text
        self.transform_label = transform_label

    def __len__(self):
        return len(self.characters_list)

    def __getitem__(self, idx):
        return self.characters_list[idx], self.texts_list[idx]

    def get_random_task(self, N=5, K=1):
        train_task, __ = self.get_random_task_split(N, train_K=K, test_K=0)
        return train_task

    def get_random_task_split(self, N=5, train_K=1, test_K=1):
        train_samples = []
        train_text_samples = []
        test_samples = []
        test_text_samples = []
        character_indices = np.random.choice(len(self), N, replace=False)
        for base_idx, idx in enumerate(character_indices):
            paths = self.characters_list[idx] # list of strings
            texts = self.texts_list[idx]
            if test_K == -1:
                curr_idxs = np.arange(len(paths))
                np.random.shuffle(curr_idxs)
            else:
                curr_idxs = np.random.choice(len(paths), train_K + test_K, replace=False) # replace=True yields bug
            for i, path_idx in enumerate(curr_idxs):
                path = paths[path_idx]
                path = {'character_idx':idx, 'path':path}
                new_path = {}
                new_path.update(path)
                new_path['base_idx'] = base_idx
                text = texts[path_idx]
                new_text = {'text':text, 'base_idx':base_idx}
                if i < train_K:
                    train_samples.append(new_path)
                    train_text_samples.append(new_text)
                else:
                    test_samples.append(new_path)
                    test_text_samples.append(new_text)
        new_train_idxs = np.arange(len(train_samples))
        np.random.shuffle(new_train_idxs)
        train_samples = [train_samples[i] for i in new_train_idxs]
        train_text_samples = [train_text_samples[i] for i in new_train_idxs]
        new_test_idxs = np.arange(len(test_samples))
        np.random.shuffle(new_test_idxs)
        test_samples = [test_samples[i] for i in new_test_idxs]
        test_text_samples = [test_text_samples[i] for i in new_test_idxs]
        train_task = FewShot(train_samples, train_text_samples,
                            meta={'characters': character_indices, 'split': 'train'},
                            parent=self
                            )
        test_task = FewShot(test_samples, test_text_samples,
                             meta={'characters': character_indices, 'split': 'test'},
                             parent=self
                             )
        return train_task, test_task
    
    def get_task_split(self, character_indices, all_curr_idxs, new_train_idxs, new_test_idxs, train_K=1):
        train_samples = []
        train_text_samples = []
        test_samples = []
        test_text_samples = []
        for base_idx, idx in enumerate(character_indices):
            paths = self.characters_list[idx] # list of strings
            texts = self.texts_list[idx]

            curr_idxs = all_curr_idxs[base_idx]
            
            for i, path_idx in enumerate(curr_idxs):
                path = paths[path_idx]
                path = {'character_idx':idx, 'path':path}
                new_path = {}
                new_path.update(path)
                new_path['base_idx'] = base_idx
                text = texts[path_idx]
                new_text = {'text':text, 'base_idx':base_idx}
                if i < train_K:
                    train_samples.append(new_path)
                    train_text_samples.append(new_text)
                else:
                    test_samples.append(new_path)
                    test_text_samples.append(new_text)
        train_samples = [train_samples[i] for i in new_train_idxs]
        train_text_samples = [train_text_samples[i] for i in new_train_idxs]
        test_samples = [test_samples[i] for i in new_test_idxs]
        test_text_samples = [test_text_samples[i] for i in new_test_idxs]
        train_task = FewShot(train_samples, train_text_samples,
                            meta={'characters': character_indices, 'split': 'train'},
                            parent=self
                            )
        test_task = FewShot(test_samples, test_text_samples,
                             meta={'characters': character_indices, 'split': 'test'},
                             parent=self
                             )
        return train_task, test_task


class MetaOmniglotFolder(AbstractMetaOmniglot):

    def __init__(self, *args, **kwargs):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(parent_dir, 'data')
        fid_to_label = load_pkl(os.path.join(data_dir, 'fid_to_label.pkl'))
        fid_to_text = load_pkl(os.path.join(data_dir, 'fid_to_text.pkl'))
        fids_path = os.path.join(data_dir, 'fids.pkl')
        label_to_int_path = os.path.join(data_dir, 'label_to_int.pkl')
        fids = load_pkl(fids_path)
        label_to_int = load_pkl(label_to_int_path)
        paths = [os.path.join(data_dir, 'images', 'img%s.jpg' % fid) for fid in fids]
        targets = [label_to_int[fid_to_label[fid]] for fid in fids]
        num_labels = len(np.unique(targets))
        grouped_paths = [[] for _ in range(num_labels)]
        for path, label in zip(paths, targets):
            grouped_paths[label].append(path)
        grouped_texts = [[] for _ in range(num_labels)]
        for fid, label in zip(fids, targets):
            grouped_texts[label].append(fid_to_text[fid])
        characters_list = []
        for i, paths in enumerate(grouped_paths):
            characters_list.append(paths)
        characters_list = np.array(characters_list)
        texts_list = []
        for i, texts in enumerate(grouped_texts):
            texts_list.append(texts)
        texts_list = np.array(texts_list)
        AbstractMetaOmniglot.__init__(self, characters_list, texts_list, *args, **kwargs)


class MetaOmniglotSplit(AbstractMetaOmniglot):
    pass


def split_omniglot(meta_omniglot, validation=0.1, test=0.1):
    '''
    Split meta-omniglot into two meta-datasets of tasks (disjoint characters)
    '''
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    split_idxs_path = os.path.join(data_dir, 'split_idxs.npy')
    indices = np.load(split_idxs_path)
    n_val = int(validation * len(meta_omniglot))
    n_test = int(test * len(meta_omniglot))
    all_train = [meta_omniglot[i] for i in indices[:-(n_val+n_test)]]
    all_val = [meta_omniglot[i] for i in indices[-(n_val+n_test):-n_test]]
    all_test = [meta_omniglot[i] for i in indices[-n_test:]]
    train_characters = [e[0] for e in all_train]
    val_characters = [e[0] for e in all_val]
    test_characters = [e[0] for e in all_test]
    train_texts = [e[1] for e in all_train]
    val_texts = [e[1] for e in all_val]
    test_texts = [e[1] for e in all_test]
    train = MetaOmniglotSplit(train_characters, train_texts, cache=meta_omniglot.cache, size=meta_omniglot.size,
                              transform_image=meta_omniglot.transform_image, transform_label=meta_omniglot.transform_label)
    val = MetaOmniglotSplit(val_characters, val_texts, cache=meta_omniglot.cache, size=meta_omniglot.size,
                             transform_image=meta_omniglot.transform_image, transform_label=meta_omniglot.transform_label)
    test = MetaOmniglotSplit(test_characters, test_texts, cache=meta_omniglot.cache, size=meta_omniglot.size,
                             transform_image=meta_omniglot.transform_image, transform_label=meta_omniglot.transform_label)
    return train, val, test


# Default transforms
transform_image = transforms.Compose([
    lambda x: torch.tensor(x).float()
    # transforms.ToTensor()
])