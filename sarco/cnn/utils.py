import hickle as hkl
import random
import numpy as np
import theano
import lasagne
import pdb

random.seed(123)

def load(e = "frontal"):
    assert e in ["frontal", "lateral"]
    frontal, lateral, y, ymm = hkl.load(open("/home/mesnilgr/repos/sarco/sarco/cnn/l3.hkl")) 

    if e == "frontal":
        frontal = frontal.astype(theano.config.floatX) - frontal.max()
        train_x, train_y, train_ymm = frontal[:128], y[:128], ymm[:128] 
        test_x, test_y, test_ymm = frontal[128:], y[128:], ymm[128:] 
    if e == "lateral":
        lateral = lateral.astype(theano.config.floatX) - lateral.max()
        train_x, train_y, train_ymm = lateral[:128], y[:128], ymm[:128] 
        test_x, test_y, test_ymm = lateral[128:], y[128:], ymm[128:] 
    
    train_x = train_x.reshape((128, 500, 500))[:, np.newaxis, :, :]
    test_x = test_x.reshape((128, 500, 500))[:, np.newaxis, :, :]
    return (train_x, train_y, train_ymm), (test_x, test_y, test_ymm)

def crop(x, y, ymm, image_shape=500, cropsize=400):
    """ x comes as bs, 1, 500, 500 """
    center_margin = (image_shape - cropsize) / 2
    crop_xs = random.randint(0, center_margin * 2)
    crop_ys = random.randint(0, center_margin * 2)
    _x = x[:, :, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize].astype(theano.config.floatX)
    _y = y - crop_ys
    _ymm = np.array((ymm - crop_ys * ymm / y)).astype(theano.config.floatX)[:, np.newaxis]
    return _x, _y, _ymm

def sample(dataset, ens, set_x, set_y, set_ymm, cropsize, uniform=False):
    assert ens in ["train", "valid"]
    if not uniform:
        x, _, y = crop(set_x, set_y, set_ymm, cropsize=cropsize)
    else:
        # every k samples with do translation
        k, batch_size = 2, 128
        x, y = [], []
        for i in range(batch_size / k):
            _x, _, _y = crop(set_x[i * k: (i + 1) * k], set_y[i * k: (i + 1) * k], set_ymm[i * k: (i + 1) * k], cropsize=cropsize)
            x += [_x]
            y += [_y]
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y)
    dataset["X_" + ens].set_value(x.astype(theano.config.floatX))
    dataset["y_" + ens].set_value(y.astype(theano.config.floatX))
 
def get_data(ens, cropsize):
    (train_x, train_y, train_ymm), (test_x, test_y, test_ymm) = load(ens)
    _train_x, _, _train_y = crop(train_x, train_y, train_ymm, cropsize=cropsize)
    _test_x, _, _test_y = crop(test_x, test_y, test_ymm, cropsize=cropsize)
    return dict(
        X_train=theano.shared(lasagne.utils.floatX(_train_x)),
        y_train=theano.shared(lasagne.utils.floatX(_train_y)),
        X_valid=theano.shared(lasagne.utils.floatX(_test_x)),
        y_valid=theano.shared(lasagne.utils.floatX(_test_y)))


