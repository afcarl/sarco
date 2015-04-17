"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import pdb
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

import matplotlib as M
M.use("Agg")
from PIL import Image as Im

from logistic_sgd import LogisticRegression, load_data, rotate_data


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.nhid = n_out
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
        #self.output = T.switch(lin_output<0, 0, lin_output)
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        cur_in, cur_in_d = input, n_in
        self.layers = []
        for n_hid in n_hidden:
            self.layers += [HiddenLayer(
                rng=rng,
                input=cur_in,
                n_in=cur_in_d,
                n_out=n_hid,
                activation=T.tanh
            )]
            cur_in = self.layers[-1].output
            cur_in_d = self.layers[-1].nhid

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            rng=rng,
            input=self.layers[-1].output,
            n_in=n_hidden[-1],
            n_out=n_out,
            activation='sigmoid'
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.logRegressionLayer.W).sum()
        for layer in self.layers:
            self.L1 += abs(layer.W).sum()
        

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()
        for layer in self.layers:
            self.L2_sqr += (layer.W ** 2).sum()
        

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.y_pred = self.logRegressionLayer.y_pred
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.logRegressionLayer.params
        for layer in self.layers:
            self.params += layer.params 
        # end-snippet-3


def test_mlp(learning_rate=0.05, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             split=0, batch_size=1, n_hidden=[100], rot = 5):
    datasets = load_data(split)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] #/ batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] #/ batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)
    shp = train_set_x.get_value().shape[1]
    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=shp,
        n_hidden=n_hidden,
        n_out=shp
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    pred_test = theano.function(
        inputs=[index],
        outputs=[classifier.y_pred, y],
        givens={
            x: test_set_x[index : (index + 1) ],
            y: test_set_y[index : (index + 1) ]
        }
    )

    pred_train = theano.function(
        inputs=[index],
        outputs=[classifier.y_pred, y],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    pred_valid = theano.function(
        inputs=[index],
        outputs=[classifier.y_pred, y],
        givens={
            x: valid_set_x[index: (index + 1) ],
            y: valid_set_y[index: (index + 1) ]
        }
    )

    def jaccard(pred, true, x, seuil=0.25, plot=False):
        if plot:
            bigpic = []
        Ms = []
        assert pred.shape[0] == true.shape[0]
        assert pred.shape[1] == true.shape[1]
        for i in range(pred.shape[0]):
            if plot:
                bigpic += [x[i], pred[i], true[i]]
            pred[i] *= (x[i] > 0).astype(numpy.float)
            M11 = (((pred[i] >= seuil).astype(numpy.int) + (true[i] == 1).astype(numpy.int)) == 2).sum()
            if M11 == 0:
                print 'no intersection between prediction and label!'
                #raise NotImplementedError 
            M10 = (((pred[i] >= seuil).astype(numpy.int) + (true[i] == 0).astype(numpy.int)) == 2).sum()
            M01 = (((pred[i] < seuil).astype(numpy.int) + (true[i] == 1).astype(numpy.int)) == 2).sum()
            #print 'M11',  M11, 'M01', M01, 'M10', M10
            Ms += [float(M11) / (M11 + M10 + M01)]
        bigpic = numpy.vstack(bigpic)
        return numpy.mean(Ms)

    def eval_train(i):
        pred, true = pred_train(i)
        return jaccard(pred, true, train_set_x.get_value())  

    def test_model(i):
        pred, true = pred_test(i)
        return jaccard(pred, true, test_set_x.get_value())  
    
    def validate_model(i):
        pred, true = pred_valid(i)
        return jaccard(pred, true, valid_set_x.get_value())

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 1# 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = -numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    
    train_losses = [eval_train(i) for i
                             in xrange(n_train_batches)]
    this_train_loss = numpy.mean(train_losses)
    #print this_train_loss

    while (epoch < n_epochs) and (not done_looping):
        rotate_data((train_set_x, train_set_y), rot)
        epoch = epoch + 1
        minibatch_avg_cost = []
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost += [train_model(minibatch_index)]


            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                print "mean avg cost over training :: ", numpy.mean(minibatch_avg_cost)
                train_losses = [eval_train(i) for i
                                         in xrange(n_train_batches)]
                this_train_loss = numpy.mean(train_losses)
                
                print(
                    'epoch %i, minibatch %i/%i, train error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_train_loss * 100.
                    )
                )

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss > best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss > best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp(learning_rate=0.001, L1_reg=0.00, L2_reg=0.00, n_epochs=1000,
             split=2, batch_size=10, n_hidden=[1024, 1024], rot=0)
 
