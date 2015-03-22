import pdb
import numpy as np
import theano
from theano.tensor.nnet import conv
import theano.tensor as T
from theano.tensor.signal import downsample

class Layer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
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

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

class ConvNet(object):
    ''' convnet model '''
    def __init__(self, rng, input, batch_size, im_shape, params):
    
    def __init__(self, rng, input, filter_shape, image_shape, poolsize):
        layers, in_shape = [], (batch_size, 1, im_shape[0], im_shape[1])
        cur_input = input
        for p in params:
             layer = LeNetConvPoolLayer(
                rng,
                input=cur_input,
                image_shape=in_shape,
                filter_shape=(nkerns[0], 1, 5, 5),
                poolsize=(2, 2)
             )

       
        pass
    def load(self):
        pass
    def save(self):
        pass
    def predict_raw(self):
        '''perform the prediction given a raw image'''
        pass


if __name__ == "__main__":
    batch_size = 16
    im_shape = (512, 512)
    params = [{"filter":(3, 3), "pool":(2, 2), "nkern":5}]
    
    sample = np.random.uniform(-1, 1, (batch_size, im_shape[0], im_shape[1])).astype(np.float32)
    label = np.zeros((batch_size, im_shape[0], im_shape[1])).astype(np.uint8)


    #learning_rate=0.1, n_epochs=200,
    #                dataset='mnist.pkl.gz',
    #                nkerns=[20, 50], batch_size=500


    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)
    pdb.set_trace()
