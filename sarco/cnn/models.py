from theano import tensor as T
import theano
import cPickle
import numpy as np
from layers import HiddenLayer, LeNetConvPoolLayer

class base(object):
    def save(self, filename):
        with open(filename, "w") as f:
            for layer in self.layers:
                for p in layer.params:
                    cPickle.dump(p.get_value(), f)

    def load(self, filename):
        with open(filename) as f:
            for layer in self.layers:
                for p in layer.params:
                    p.set_value(cPickle.load(f))

class CNN4Layers(base):
    def __init__(self, cropsize, batch_size, nkerns=[10, 10, 10, 10], filters=[11, 6, 4, 3]):
        self.X_batch, self.y_batch = T.tensor4('x'), T.matrix('y')
        self.layers, self.params = [], []
        rng = np.random.RandomState(23455)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=self.X_batch,
            image_shape=(batch_size, 1, cropsize, cropsize),
            filter_shape=(nkerns[0], 1, filters[0], filters[0]),
            poolsize=(2, 2)
        )
        self.layers += [layer0]
        self.params += layer0.params
        # 400 - 11 + 1 = 390 / 2 = 195
        map_size = (cropsize - filters[0] + 1) / 2
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], map_size, map_size),
            filter_shape=(nkerns[1], nkerns[0], filters[1], filters[1]),
            poolsize=(2, 2)
        )
        self.layers += [layer1]
        self.params += layer1.params

        # 195 - 6 + 1 = 190 / 2 = 95
        map_size = (map_size - filters[1] + 1) / 2
        layer2 = LeNetConvPoolLayer(
            rng,
            input=layer1.output,
            image_shape=(batch_size, nkerns[1], map_size, map_size),
            filter_shape=(nkerns[2], nkerns[1], filters[2], filters[2]),
            poolsize=(2, 2)
        )
        self.layers += [layer2]
        self.params += layer2.params
 
        # 95 - 4 + 1 = 92 / 2 = 46
        map_size = (map_size - filters[2] + 1) / 2
        layer3 = LeNetConvPoolLayer(
            rng,
            input=layer2.output,
            image_shape=(batch_size, nkerns[2], map_size, map_size),
            filter_shape=(nkerns[3], nkerns[2], filters[3], filters[3]),
            poolsize=(2, 2)
        )
        self.layers += [layer3]
        self.params += layer3.params
 

        # 46 - 3 + 1 = 44 / 2 = 22
        map_size = (map_size - filters[3] + 1) / 2
        layer4_input = layer3.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer4 = HiddenLayer(
            rng,
            input=layer4_input,
            n_in=nkerns[3] * map_size * map_size,
            n_out=1,
            activation=None
        )
        self.layers += [layer4]
        self.params += layer4.params

        nparams = np.sum([p.get_value().flatten().shape[0]
                          for p in self.params])
        print "model contains %i parameters" % nparams
        self.output = self.layers[-1].output

class CNN3Layers(base):
    def __init__(self, cropsize, batch_size, nkerns=[10, 10, 10], filters=[11, 6, 4]):
        self.X_batch, self.y_batch = T.tensor4('x'), T.matrix('y')
        self.layers, self.params = [], []
        rng = np.random.RandomState(23455)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=self.X_batch,
            image_shape=(batch_size, 1, cropsize, cropsize),
            filter_shape=(nkerns[0], 1, filters[0], filters[0]),
            poolsize=(2, 2)
        )
        self.layers += [layer0]
        self.params += layer0.params
        # 400 - 11 + 1 = 390 / 2 = 195
        map_size = (cropsize - filters[0] + 1) / 2
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], map_size, map_size),
            filter_shape=(nkerns[1], nkerns[0], filters[1], filters[1]),
            poolsize=(2, 2)
        )
        self.layers += [layer1]
        self.params += layer1.params

        # 195 - 6 + 1 = 190 / 2 = 95
        map_size = (map_size - filters[1] + 1) / 2
        layer2 = LeNetConvPoolLayer(
            rng,
            input=layer1.output,
            image_shape=(batch_size, nkerns[1], map_size, map_size),
            filter_shape=(nkerns[2], nkerns[1], filters[2], filters[2]),
            poolsize=(2, 2)
        )
        self.layers += [layer2]
        self.params += layer2.params
 
        # 95 - 4 + 1 = 92 / 2 = 46
        map_size = (map_size - filters[2] + 1) / 2
        layer3_input = layer2.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer3 = HiddenLayer(
            rng,
            input=layer3_input,
            n_in=nkerns[2] * map_size * map_size,
            n_out=1,
            activation=None
        )
        self.layers += [layer3]
        self.params += layer3.params

        nparams = np.sum([p.get_value().flatten().shape[0]
                          for p in self.params])
        print "model contains %i parameters" % nparams
        self.output = self.layers[-1].output
 
class CNN2Layers(base):
    def __init__(self, cropsize, batch_size, nkerns=[10, 10], filters=[11, 6]):
        self.X_batch, self.y_batch = T.tensor4('x'), T.matrix('y')
        self.layers, self.params = [], []
        rng = np.random.RandomState(23455)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=self.X_batch,
            image_shape=(batch_size, 1, cropsize, cropsize),
            filter_shape=(nkerns[0], 1, filters[0], filters[0]),
            poolsize=(2, 2)
        )
        self.layers += [layer0]
        self.params += layer0.params
        # 400 - 11 + 1 = 390 / 2 = 195
        # 300 - 11 + 1 = 290 / 2 = 145
        map_size = (cropsize - filters[0] + 1) / 2
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], map_size, map_size),
            filter_shape=(nkerns[1], nkerns[0], filters[1], filters[1]),
            poolsize=(2, 2)
        )
        self.layers += [layer1]
        self.params += layer1.params
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        map_size = (map_size - filters[1] + 1) / 2
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * map_size * map_size,
            n_out=1,
            activation=None
        )
        self.layers += [layer2]
        self.params += layer2.params

        nparams = np.sum([p.get_value().flatten().shape[0]
                          for p in self.params])
        print "model contains %i parameters" % nparams
        self.output = self.layers[-1].output
 
class CNN3Layers1Dense(base):
    def __init__(self, cropsize, batch_size, nkerns=[10, 10, 10], filters=[11, 6, 4]):
        self.X_batch, self.y_batch = T.tensor4('x'), T.matrix('y')
        self.layers, self.params = [], []
        rng = np.random.RandomState(23455)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=self.X_batch,
            image_shape=(batch_size, 1, cropsize, cropsize),
            filter_shape=(nkerns[0], 1, filters[0], filters[0]),
            poolsize=(2, 2)
        )
        self.layers += [layer0]
        self.params += layer0.params
        # 400 - 11 + 1 = 390 / 2 = 195
        map_size = (cropsize - filters[0] + 1) / 2
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], map_size, map_size),
            filter_shape=(nkerns[1], nkerns[0], filters[1], filters[1]),
            poolsize=(2, 2)
        )
        self.layers += [layer1]
        self.params += layer1.params

        # 195 - 6 + 1 = 190 / 2 = 95
        map_size = (map_size - filters[1] + 1) / 2
        layer2 = LeNetConvPoolLayer(
            rng,
            input=layer1.output,
            image_shape=(batch_size, nkerns[1], map_size, map_size),
            filter_shape=(nkerns[2], nkerns[1], filters[2], filters[2]),
            poolsize=(2, 2)
        )
        self.layers += [layer2]
        self.params += layer2.params
 
        # 95 - 4 + 1 = 92 / 2 = 46
        map_size = (map_size - filters[2] + 1) / 2
        layer3_input = layer2.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer3 = HiddenLayer(
            rng,
            input=layer3_input,
            n_in=nkerns[2] * map_size * map_size,
            n_out=100,
            activation=None
        )
        self.layers += [layer3]
        self.params += layer3.params

        layer4 = HiddenLayer(
            rng,
            input=layer3.output,
            n_in=100,
            n_out=1,
            activation=None
        )
        self.layers += [layer4]
        self.params += layer4.params


        nparams = np.sum([p.get_value().flatten().shape[0]
                          for p in self.params])
        print "model contains %i parameters" % nparams
        self.output = self.layers[-1].output

class CNN2Layers1Dense(base):
    def __init__(self, cropsize, batch_size, nkerns=[10, 10], filters=[11, 6]):
        self.X_batch, self.y_batch = T.tensor4('x'), T.matrix('y')
        self.layers, self.params = [], []
        rng = np.random.RandomState(23455)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=self.X_batch,
            image_shape=(batch_size, 1, cropsize, cropsize),
            filter_shape=(nkerns[0], 1, filters[0], filters[0]),
            poolsize=(2, 2)
        )
        self.layers += [layer0]
        self.params += layer0.params
        # 400 - 11 + 1 = 390 / 2 = 195
        # 300 - 11 + 1 = 290 / 2 = 145
        map_size = (cropsize - filters[0] + 1) / 2
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], map_size, map_size),
            filter_shape=(nkerns[1], nkerns[0], filters[1], filters[1]),
            poolsize=(2, 2)
        )
        self.layers += [layer1]
        self.params += layer1.params
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        map_size = (map_size - filters[1] + 1) / 2
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * map_size * map_size,
            n_out=1000,
            activation=None
        )
        self.layers += [layer2]
        self.params += layer2.params

        layer3 = HiddenLayer(
            rng,
            input=layer2.output,
            n_in=1000,
            n_out=1,
            activation=None
        )
        self.layers += [layer3]
        self.params += layer3.params


        nparams = np.sum([p.get_value().flatten().shape[0]
                          for p in self.params])
        print "model contains %i parameters" % nparams
        self.output = self.layers[-1].output


def theano_fns(model, dataset, learning_rate=0):
    cost = T.mean((model.output - model.y_batch) ** 2)
    error = abs(model.output - model.y_batch)
    grads = T.grad(cost, model.params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(model.params, grads)
    ]

    train_fn = theano.function([], [cost, error], updates=updates,
            givens={model.X_batch: dataset['X_train'],
                    model.y_batch: dataset['y_train']}) 

    valid_fn = theano.function([], [error, model.output],
            givens={model.X_batch: dataset['X_valid'],
                    model.y_batch: dataset['y_valid']}) 
    return train_fn, valid_fn


