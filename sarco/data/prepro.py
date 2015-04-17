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

def mask(x, a, b):
    inf = (x > a).astype(np.int32)
    sup = (x < b).astype(np.int32)
    y = (inf + sup) == 2
    y = (y * 255.).astype(np.uint8)
    return y

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
     
     
def scale_to_unit_interval(ndar, eps=1e-8):
  """ Scales all values in the ndarray ndar to be between 0 and 1 """
  ndar = ndar.copy()
  ndar -= ndar.min()
  ndar *= 1.0 / (ndar.max() + eps)
  return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=False,
                       output_pixel_vals=False):
  """
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.

  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).

  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.

  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image

  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)

  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats

  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not


  :returns: array suitable for viewing as an image.
  (See:`Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.

  """

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2

  # The expression below can be re-written in a more C style as
  # follows :
  #
  # out_shape = [0,0]
  # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
  #                tile_spacing[0]
  # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
  #                tile_spacing[1]
  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
      assert len(X) == 4
      # Create an output np ndarray to store the image
      if output_pixel_vals:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
      else:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

      #colors default to 0, alpha defaults to 1 (opaque)
      if output_pixel_vals:
          channel_defaults = [0, 0, 0, 255]
      else:
          channel_defaults = [0., 0., 0., 1.]

      for i in xrange(4):
          if X[i] is None:
              # if channel is None, fill it with zeros of the correct
              # dtype
              out_array[:, :, i] = np.zeros(out_shape,
                      dtype='uint8' if output_pixel_vals else out_array.dtype
                      ) + channel_defaults[i]
          else:
              # use a recurrent call to compute the channel and store it
              # in the output
              out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
      return out_array

  else:
      # if we are dealing with only one channel
      H, W = img_shape
      Hs, Ws = tile_spacing
      # generate a matrix to store the output
      out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


      for tile_row in xrange(tile_shape[0]):
          for tile_col in xrange(tile_shape[1]):
              if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                  if scale_rows_to_unit_interval:
                      # if we should scale values to be between 0 and 1
                      # do this by calling the `scale_to_unit_interval`
                      # function
                      this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                  else:
                      this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                  # add the slice to the corresponding position in the
                  # output array
                  out_array[
                      tile_row * (H+Hs): tile_row * (H + Hs) + H,
                      tile_col * (W+Ws): tile_col * (W + Ws) + W
                      ] \
                      = this_img * (255 if output_pixel_vals else 1)
      return out_array

