import numpy as np
import cv2
from io import StringIO
from scipy import ndimage
import sys

files = ['obj_height']
# files = ['dem', 'slope', 'aspect', 'band_1', 'band_2', 'band_3', 'band_4', 'ndvi', 'footprints', 'obj_height']
# files = ['band_2', 'band_3', 'band_4', 'ndvi', 'footprints', 'slope', 'evi', 'obj_height']
# special_layers = ['footprints', 'obj_height']
# files = ['SW_Orinda_Fat/dem', 'OrindaHome/dem', 'OrindaDowns/dem', 'SW_Orinda_Skinny/dem', 'Test_Site/dem']

# folder = '/Users/dtradke/Documents/Waterloo/CS_886_Theory_of_Deep_Learning/cnn_veg_height/data/'
folder = "../data/" + sys.argv[-1] + "/"

for f in files:
    print(f)
    # if f in special_layers:
    #     if f == 'footprints':
    #         fname = folder + 'special_layers/'+ f + '.tif'
    #         layer = cv2.imread(fname, cv2.IMREAD_COLOR)
    #         the_type = layer.dtype
    #         layer = layer[:,:,0]
    #         new_fname = folder + 'special_layers/'+ f + '.txt'
    #     else:
    #         fname = folder + 'special_layers/'+ f + '.tif'
    #         layer = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    #         the_type = layer.dtype
    #         new_fname = folder + 'special_layers/'+ f + '.txt'
    # else:

    float_formatter = lambda x: "%f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})

    fname = folder + f + '.tif'
    print(fname)
    if f == 'footprints':
        print("IMREAD_COLOR")
        layer = cv2.imread(fname, cv2.IMREAD_COLOR)
        layer = layer[:,:,0]
    else:
        layer = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

    print(layer.shape)
    the_type = layer.dtype
    new_fname = folder + f + '.txt'
    # print(layer)

    # if f == 'obj_height':
    # print('OBJ_HEIGHT')
    layer[layer<0] = 0
    # layer[layer>250] = 250
    mask = np.ones((3, 3))
    # mask[1, 1] = 0
    layer = ndimage.generic_filter(layer, np.nanmean, footprint=mask, mode='constant', cval=np.NaN)



        # print(layer.shape)
        # # print(new_fname)
        # exit()
    np.savetxt(new_fname, layer, delimiter=',')
    print(fname)
    print(the_type)




#below is for loadFootprints

# f = 'footprints'
# fname = folder + f + '.tif'
# print(fname)
# layer = cv2.imread(fname, cv2.IMREAD_COLOR)
# layer = layer[:,:,0]
# print(layer.shape)
# the_type = layer.dtype
# new_fname = folder + f + '.txt'
