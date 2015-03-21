import pdb
import numpy as np

class ConvNet(object):
    ''' convnet model '''
    def __init__(self):
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
    sample = np.random.uniform(-1, 1, (batch_size, 512, 512)).astype(np.float32)
    label = np.zeros((batch_size, 512, 512)).astype(np.uint8)
    pdb.set_trace()
