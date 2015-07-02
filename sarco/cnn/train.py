import pdb
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

    train_costs, train_errors, naive_train = [], [] , []
    best_val = np.inf 
    for i in range(10000):
        # sample a new dataset
        sample(dataset, "train", train_x, train_y, train_ymm, cropsize, uniform=True)
        
        # compute naive prediction
        naive_train += [dataset['y_train'].get_value()]
        
        # train
        train_cost, train_error = train_fn()
        train_costs += [train_cost]
        train_errors += [train_error]

        # compute validation error
        if (i + 1) % val_frequency == 0:
            valid_error, naive_error = [], []
            naive_pred = np.mean(naive_train)
            
            for j in range(n_batches_valid):
                # sample a new dataset
                sample(dataset, "valid", test_x, test_y, test_ymm, cropsize)

                naive_err = abs(dataset['y_valid'].get_value() - naive_pred)
                valid_err, out = valid_fn()
                valid_error += [valid_err.T]
                naive_error += [naive_err.T]
            
            valid_error = np.hstack(valid_error)
            naive_error = np.hstack(naive_error)
            
            # early stopping
            if np.mean(valid_error) < best_val:
                best_val = np.mean(valid_error)
                best_val_std = np.std(valid_error)
                model.save("models/best-%s-%s-%icrop.pkl" % (ens, modelname, cropsize))

            print "update %i train cost :: %.2f \t train_error :: %.2f (in mm) \t \
                valid error is %.2f +- %.2f (in mm) (best is %.2f +- %.2f) \t naive error is %.2f +- %.2f (in mm)" \
                % ((i + 1), np.mean(train_costs), \
                np.mean(train_errors), np.mean(valid_error), \
                np.std(valid_error), best_val, best_val_std, np.mean(naive_error), np.std(naive_error))

            train_costs, train_errors = [], [] 
            # save model
