'''
Based on https://github.com/gabrielhuang/reptile-pytorch

Peter Wu
peterw1@andrew.cmu.edu
'''

import numpy as np
import os
import pickle
import re

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix


def save_pkl(obj, path):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def mk_label_map(label_set):
    '''
    Returns dictionary mapping raw label to integer (0-indexed)
    '''
    label_to_int = {}
    for i, label in enumerate(label_set):
        label_to_int[label] = i
    return label_to_int


def transform_label(paths):
    return paths['base_idx']


# Those two functions are taken from torchvision code because they are not available on pip as of 0.2.0
def list_dir(root, prefix=False):
    '''List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    '''
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    '''List all files ending with a suffix at a given root
    
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    '''
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def find_latest_file(folder):
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        return max(files)[1]
    else:
        return None

pass


def svm_classify(train_data, test_data, train_label, test_label, C=0.01):
    '''trains a linear SVM on the data

    Args:
        train_data: np array with shape (batch_size, fc_dim)
        C: specifies the penalty factor of SVM
    '''
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    test_conf_mat = confusion_matrix(test_label, p)
    np.save('test_conf_mat.npy', test_conf_mat)

    return test_acc