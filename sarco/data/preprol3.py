import dicom
"https://code.google.com/p/pydicom/wiki/ViewingImages"
from PIL import Image as Im
import os
from os.path import join
from collections import Counter
import pdb
import pylab as P
import numpy as np
from sarco.data.prepro import seuil, to01, mask, tile_raster_images

PATH = "/Users/gregoire/Desktop/BDD_SARCO_V1"
shp, origin, end = Counter(), Counter(), Counter()

def isdicom(fname):
    fname = fname.lower()
    if '.' in fname:
        if fname.split('.')[-1] == "dcm":
            return True
    return False

patient = 0
image = 0
plot = True
if plot:
    center = np.zeros((512, 512))

o3000, o0 = [], []

def show_mask(im, a, b):
    Im.fromarray(mask(im, a, b)).show()

box = np.zeros((512, 512))
shift = 200
minima = 10

bbox = (126, 190, 386, 450)
print bbox[2] - bbox[0], bbox[3] - bbox[1]
bigpic = []

for d in os.listdir(PATH):
    subpath = join(PATH, d)
    
    if os.path.isfile(subpath):
        continue

    patient +=1
    subpath = join(PATH, d, 'coupeL3')
    for im in os.listdir(subpath):
        impath = join(subpath, im) 
        if isdicom(impath):
            #if image > 500: break
            ds = dicom.read_file(impath)
            im = ds.pixel_array.astype(np.float32)

            # checks shape
            try: 
                assert im.shape[0] == 512 and im.shape[1] == 512
            except AssertionError:
                print "skipping", impath, "shape is not (512, 512) but ", im.shape
                continue
            
            if im.min() > -1:
                m_im = mask(im, shift + 1000, 5000)
            else:
                m_im = mask(im, shift, 5000)
             
            # check threshold does not explode
            try:
                assert m_im.mean() < 6
            except AssertionError:
                print "skipping", impath, "threshold exploded [0, 6] ", m_im.mean()
                continue

            box += m_im
            if m_im.min() < minima:
                minima = m_im.min()

            bigpic += [np.array(Im.fromarray(m_im).crop(bbox)).flatten()]
            image += 1

            #if plot and im.shape[0] == 512 and im.shape[1] == 512:
            #    if not np.isnan(im).any():
            #        center += im
            shp["-".join(map(str, im.shape))] += 1
            origin[str(im.min())] += 1
            end[str(im.max())] += 1
            
#bbox = ((box > 0) * 255.).astype(np.uint8)
#Im.fromarray(bbox).show()
#Im.fromarray(bbox).save("box.png")
bigpic = np.vstack(bigpic)
tile = tile_raster_images(X=bigpic, img_shape=(260, 260), tile_shape=(17, 17))
Im.fromarray(tile).save("bigpic.png")

print shp
print origin
print end
print patient, " #patients in total"
print image, " #images in total"
print "minimum avergae after threshold", minima



