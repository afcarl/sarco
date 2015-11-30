import pdb
import sys
import time
import numpy as np
from theano import tensor as T
import theano
import cPickle

from utils import load, crop, sample, get_data
from layers import HiddenLayer, LeNetConvPoolLayer
from models import CNN4Layers, CNN3Layers, CNN2Layers
from models import CNN3Layers1Dense, theano_fns

if __name__ == "__main__":

    debug = False
    ens = "frontal"
    cropsize = 400
    batch_size = 128
    learning_rate = 0.00001
    val_frequency = 100
    n_batches_valid = 100
    nlayers = 3

    if debug:
        val_frequency = 1
        n_batches_valid = 10

    (train_x, train_y, train_ymm), (test_x, test_y, test_ymm) = load(ens)
    dataset = get_data(ens, cropsize)
    #modelname = "CNN%iLayers1Dense" % nlayers    
    modelname = "CNN%iLayers" % nlayers    
    model = eval("%s(cropsize, batch_size)" % modelname)
    train_fn, valid_fn = theano_fns(model, dataset, learning_rate)

    modelpath = "models/best-%s-%s-%icrop.pkl" % (ens, modelname, cropsize)
    print "loading", modelpath
    model.load(modelpath)
   
    valid_error, naive_error = [], []
    
    for j in range(n_batches_valid):
        print "%i/%i\r" % (j + 1, n_batches_valid),
        sys.stdout.flush()
        # sample a new dataset
        sample(dataset, "valid", test_x, test_y, test_ymm, cropsize)
        valid_err, out = valid_fn()
        valid_error += [valid_err.T]

    valid_error = np.hstack(valid_error)
   
    print "%.2f +- %.2f error in mm" % (np.mean(valid_error), np.std(valid_error))


