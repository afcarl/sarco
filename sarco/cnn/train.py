import hickle as hkl
import pdb
import lasagne
import theano
from theano import tensor as T
import time
import numpy as np
import random
import numpy
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

""" 
TODO have different crop inside the same batch
"""

def load(e = "frontal"):
    assert e in ["frontal", "lateral"]
    frontal, lateral, y, ymm = hkl.load(open("/home/mesnilgr/repos/sarco/sarco/cnn/l3.hkl")) 
    #ymm /= ymm.max()

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

def crop(x, y, ymm, seed=123, image_shape=500, cropsize=400):
    """ x comes as bs, 1, 500, 500 """
    random.seed(seed)
    center_margin = (image_shape - cropsize) / 2
    crop_xs = random.randint(0, center_margin * 2)
    crop_ys = random.randint(0, center_margin * 2)
    _x = x[:, :, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize].astype(theano.config.floatX)
    _y = y - crop_ys
    _ymm = np.array((ymm - crop_ys * ymm / y)).astype(theano.config.floatX)[:, np.newaxis]
    return _x, _y, _ymm

def sample(dataset, ens, seed, train_x, train_y, train_ymm):
    assert ens in ["train", "valid"]
    x, _, y = crop(train_x, train_y, train_ymm, seed)
    dataset["X_" + ens].set_value(x.astype(theano.config.floatX))
    dataset["y_" + ens].set_value(y.astype(theano.config.floatX))


def get_data(ens):
    (train_x, train_y, train_ymm), (test_x, test_y, test_ymm) = load(ens)
    _train_x, _, _train_y = crop(train_x, train_y, train_ymm)
    _test_x, _, _test_y = crop(test_x, test_y, test_ymm)
    return dict(
        X_train=theano.shared(lasagne.utils.floatX(_train_x)),
        y_train=theano.shared(lasagne.utils.floatX(_train_y)),
        X_valid=theano.shared(lasagne.utils.floatX(_test_x)),
        y_valid=theano.shared(lasagne.utils.floatX(_test_y)))

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def build_model(input, cropsize, batch_size = 128):


    input_width, input_height = (cropsize, cropsize)
    l_in = lasagne.layers.InputLayer(shape=(
        batch_size, 1, input_width, input_height))

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=32,
        filter_size=(11, 11),
        nonlinearity=lasagne.nonlinearities.rectify)

    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2, 2))

    if False:
        l_conv2 = lasagne.layers.Conv2DLayer(
            l_pool1,
            num_filters=32,
            filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

        l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2, 2))

        l_hidden1 = lasagne.layers.DenseLayer(
            l_pool2,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify)

        l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_pool1,
        #l_hidden1,
        num_units=1,
        nonlinearity=None)

    return l_out

def build_model2(x, cropsize, batch_size, nkerns=[10, 10]):
    layer0_input = x#.reshape((batch_size, 1, cropsize, cropsize))

    rng = np.random.RandomState(23455)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, cropsize, cropsize),
        filter_shape=(nkerns[0], 1, 11, 11),
        poolsize=(2, 2)
    )
    # 400 - 11 + 1 = 390 / 2 = 195
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 195, 195),
        filter_shape=(nkerns[1], nkerns[0], 6, 6),
        poolsize=(2, 2)
    )
    # 195 - 6 + 1 = 190 / 2 = 95

    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 95 * 95,
        n_out=1,
        activation=None
    )
    params = layer2.params + layer1.params + layer0.params
    return layer2, params
 
def theano_fns(dataset, l_out, momentum=0.9):

   
    
    objective = lasagne.objectives.Objective(
        l_out,
        loss_function=lasagne.objectives.mse)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch,
                                   deterministic=True)

    #pred = T.argmax(
    #    output_layer.get_output(X_batch, deterministic=True), axis=1)
    #accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(l_out)
    _learning_rate = theano.shared(np.array(learning_rate, dtype=theano.config.floatX),
                                   name="learning_rate")
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, _learning_rate, momentum)

    iter_train = theano.function(
        [], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'],
            y_batch: dataset['y_train'],
        },
    )

    iter_valid = theano.function(
        [], loss_eval,
        givens={
            X_batch: dataset['X_valid'],
            y_batch: dataset['y_valid'],
        },
    )

    return iter_train, iter_valid, _learning_rate


def decay(lr):
    lr_value = lr.get_value()
    lr_value *= 0.1
    lr.set_value(lr_value)

if __name__ == "__main__":

    ens = "frontal"
    cropsize = 400
    batch_size = 128
    learning_rate = 0.00001
    nkerns = [10, 10]

    (train_x, train_y, train_ymm), (test_x, test_y, test_ymm) = load(ens)
    dataset = get_data(ens)
    #model = build_model(cropsize, batch_size)
    X_batch = T.tensor4('x')
    y_batch = T.matrix('y')
    model, params = build_model2(X_batch, cropsize, batch_size, nkerns)
    cost = T.mean((model.output - y_batch) ** 2)
    error = abs(model.output - y_batch)
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_fn = theano.function([], [cost, error], updates=updates,
            givens={X_batch: dataset['X_train'],
                    y_batch: dataset['y_train']}) 

    valid_fn = theano.function([], [error, model.output],
            givens={X_batch: dataset['X_valid'],
                    y_batch: dataset['y_valid']}) 

    #train_fn, valid_fn, lr = theano_fns(dataset, model)
    train_costs, train_errors = [], [] 
    for i in range(10000):
        # sample a new dataset
        sample(dataset, "train", i, train_x, train_y, train_ymm)
        train_cost, train_error = train_fn()
        train_costs += [train_cost]
        train_errors += [train_error]
        #if (i + 1) % 100 == 0:
        if (i + 1) % 100 == 0:
            valid_error = []
            tic = time.time()
            for j in range(100):
            #for j in range(10):
                sample(dataset, "valid", j, test_x, test_y, test_ymm)
                valid_err, out = valid_fn()
                valid_error += [valid_err.T]
            #print "computing valid on 10 took %.2f sec" % (time.time() - tic)
            valid_error = np.hstack(valid_error)
            print "update %i train cost :: %.2f \t train_error :: %.2f (in mm) \t valid error is %.2f +- %.2f (in mm)" % ((i + 1), np.mean(train_costs), np.mean(train_errors), np.mean(valid_error), np.std(valid_error))
            train_costs, train_errors = [], [] 

