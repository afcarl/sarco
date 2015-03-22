import cPickle, gzip
import numpy as np
import pdb
from sarco.data.prepro import load

PATH = "/Users/gregoire/Code/sarco/sarco/data/sarco_data.pkl.gz"
PATH = "/data/lisatmp2/mesnilgr/sarco/sarco/data/sarco_data.pkl.gz"

def get_split(i):
    ''' return train, valid, test for split i'''
    assert i in range(4)
    x, y, splits = load(PATH)
    train_idx = list(set(range(128)) - set(list(splits[i]) + list(splits[4])))
    train = (x[train_idx], y[train_idx])
    valid = (x[splits[i]], y[splits[i]])
    test = (x[splits[4]], y[splits[4]])
    return train, valid, test

def get_whole():
    ''' return all the training images'''
    x, y, splits = load(PATH)
    train_idx = np.hstack(splits[:4])
    train = (x[train_idx], y[train_idx])
    test = (x[splits[4]], y[splits[4]])
    return train, test

if __name__ == "__main__":
    #o = load(PATH)
    for i in range(5):
        continue
        try:
            get_split(i)
        except AssertionError:
            assert i == 4
    get_whole()

