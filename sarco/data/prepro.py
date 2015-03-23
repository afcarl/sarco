from os.path import join
from PIL import Image as Im
import pdb
import numpy as np
import pylab as P
import cPickle, gzip
import scipy.io

PATH_FOLDS = '/Users/gregoire/Downloads/5-folds.mat'
PATH = '/Users/gregoire/Downloads/sarcopenie_15_03_18/'

def save(obj, filename, protocol=-1):
    with gzip.GzipFile(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)

def load(filename):
    with gzip.GzipFile(filename, 'r') as f:
        return cPickle.load(f)

def seuil(x, a, b):
    inf = (x > a).astype(np.int32)
    sup = (x < b).astype(np.int32)
    y = (inf + sup) == 2
    x = x * y
    zeros = (1 - y)
    x += zeros * a
    return x

def to01(x):
    x = x.astype(np.float32)
    x -= x.min()
    x /= x.max()
    return x

def box(x, h, w):
    h0, w0 = x.shape
    assert h < h0 and w < w0
    h1, w1 = (h0 - h) // 2, (w0 - w) // 2
    x = x[h1:h1 + h, w1:w1 + w]
    nh, nw = x.shape
    assert nh == h and nw == w
    return x

def prepro_im(i, style):
    assert style in ['warped', 'original']
    x = Im.open(join(PATH, str(i).zfill(3), style, "l3.png"))
    x = np.array(x) - 1024 # remove shift origin HU
    x = seuil(x, -29, 150) # seuillage
    x = to01(x) # between [0, 1]
    x = box(x, 311, 457)
    return x

def prepro_label(i, style):
    x = Im.open(join(PATH, str(i).zfill(3), style, "muscle.png"))
    x = np.array(x).astype('uint8')
    x = box(x, 311, 457)
    return x

# tests

def test_seuil():
    
    def test(x, a, b):
        print a, b
        print x
        print seuil(x, a, b)
        pdb.set_trace()
    
    # neg
    x = - np.eye(3)*10
    print test(x, -11, 1)
    print test(x, -10, 0)
    print test(x, -11, -8)
 
    x = np.eye(3)*10
    print test(x, 4, 10)
    print test(x, 4, 5)
    print test(x, 4, 11)

if __name__ == '__main__':
    #test_seuil()
    raw = scipy.io.loadmat(PATH_FOLDS)['folds']
    splits = [raw[0][i][0] - 1 for i in range(5)]

    x, y = [], []
    for i in range(128):
        x += [prepro_im(i + 1, 'warped')]
        y += [prepro_label(i + 1, 'warped')]
        print i
    x, y = np.array(x), np.array(y)
    dataset = (x, y, splits)
    save(dataset, "sarco_data.pkl.gz")
     
     


