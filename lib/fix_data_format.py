import numpy as np
import cv2
from io import StringIO
from scipy import ndimage
import sys
import PIL
from PIL import Image

# files = ['obj_height']
files = ['dem', 'slope', 'aspect', 'band_1', 'band_2', 'band_3', 'band_4', 'ndvi', 'obj_height'] #, 'footprints'
# files = ['dem', 'slope', 'aspect', 'band_1', 'band_2', 'band_3', 'band_4', 'ndvi', 'obj_height', 'footprints'] #, 'footprints'


dirs = [sys.argv[-1]]

PIL.Image.MAX_IMAGE_PIXELS = 240000000

for dir in dirs:
    # folder = "../data/" + sys.argv[-1] + "/"
    folder = "../data/" + dir + "/"

    for f in files:
        print(f)

        float_formatter = lambda x: "%f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})

        fname = folder + f + '.tif'
        print(fname)

        im = Image.open(fname)
        layer = np.array(im)
        # layer = np.load(fname)

        # the_type = layer.dtype
        # layer = cv2.imread(fname, cv2.IMREAD_COLOR)
        # layer = layer[:,:,0]
        # layer[layer>0] = 1

        # layer = np.load(fname)
        print(layer.shape)
        new_fname = folder + f + '.npy' #'.txt'
        # print(layer)

        # if f == 'obj_height':
        # print('OBJ_HEIGHT')
        layer[layer<0] = 0
        # layer[layer>250] = 250

        if f == 'obj_height':
            mask = np.ones((3, 3))
        #   mask[1, 1] = 0
        #     # layer = ndimage.generic_filter(layer, np.nanmean, footprint=mask, mode='constant', cval=np.NaN)
            layer = ndimage.generic_filter(layer, np.median, footprint=mask, mode='constant', cval=np.NaN)



        # print(layer.shape)
            # # print(new_fname)
            # exit()
        # np.savetxt(new_fname, layer, delimiter=',')
        np.save(new_fname, layer)
        # print(fname)
        # print(the_type)




#below is for loadFootprints

# f = 'footprints'
# fname = folder + f + '.tif'
# print(fname)
# layer = cv2.imread(fname, cv2.IMREAD_COLOR)
# layer = layer[:,:,0]
# print(layer.shape)
# the_type = layer.dtype
# new_fname = folder + f + '.txt'
