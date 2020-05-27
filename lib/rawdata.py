from os import listdir
import os
import numpy as np
import sys
# import cv2
# from scipy import ndimage
from multiprocessing import Pool
from keras.utils import to_categorical

from lib import util
from lib import viz

PIXEL_SIZE = 1

classify = False
bin_class = False

small_obj_heights = False
AUGMENT = True

def loadLocations(input_arr):
    locName = input_arr
    return {locName:Location.load(locName, 'all')}

class RawData(object):

    def __init__(self, locs):
        self.locs = locs

        name_arr = []
        for i in self.locs.values():
            print("loc: ", i)
            name_arr.append(i.name)

        start = ''
        self.names = start.join(name_arr)
        print("RawData names: ", self.names)

    @staticmethod
    def load(locNames='all', special_layers='all', new_data=None):
        print("in rawdata load")
        if locNames == 'all':
            locNames = listdir_nohidden('data/')
        if locNames == 'untrain':
            locNames = listdir_nohidden('data/_untrained/')
        if special_layers == 'all':
            if new_data is None:
#training
                print("Loading all new datasets")
                locs = {}
                cores = 4
                chunksize = 1
                eb_train = False

                if 'East_Bay' in locNames:
                    eb_train = True
                    locNames.remove('East_Bay') #too large for multiprocessing
                print("LocName: ", locNames)

                with Pool(processes=cores) as pool:
                    location_list_return = pool.map(loadLocations, locNames, chunksize)
                if eb_train: location_list_return.append(loadLocations('East_Bay'))

                for i in location_list_return:
                    locs[list(i.keys())[0]] = i[list(i.keys())[0]]

                if AUGMENT:
                    # new_locs = {}
                    # rotations = 1
                    # while rotations < 4:
                    #     for i, key in enumerate(locs.keys()):
                    #         key_string = key + str(rotations)
                    #         specialLayers, layer_obj_heights, rot_layers = locs[key].rotate((rotations * 90))
                    #         new_locs[key_string] = Location(key_string, specialLayers, layer_obj_heights, rot_layers)
                    #     rotations+=1
                    #
                    # for i, key in enumerate(locs.keys()):
                    #     key_string = key + "_shift_"+ str(rotations)
                    #     specialLayers, layer_obj_heights, rot_layers = locs[key].shift()
                    #     new_locs[key_string] = Location(key_string, specialLayers, layer_obj_heights, rot_layers)
                    # locs.update(new_locs)
                    # print(locs)

                    # -- only flip
                    new_locs = {}
                    rotations = 2
                    for i, key in enumerate(locs.keys()):
                        if key != 'East_Bay':
                            key_string = key + str(rotations)
                            specialLayers, layer_obj_heights, rot_layers = locs[key].rotate((rotations * 90))
                            new_locs[key_string] = Location(key_string, specialLayers, layer_obj_heights, rot_layers)

                    locs.update(new_locs)
                    print(locs)
#endtraining
            else:
                print('one testing location')
                locs = {n:Location.load(n, 'all') for n in locNames}

        else:
            # assumes dates is a dict, with keys being locNames and vals being special_layers
            locs = {n:Location.load(n, special_layers[n]) for n in locNames}
        return RawData(locs)


    def getClassificationOutput(self, locName, location):
        loc = self.locs[locName]
        return loc.obj_height_classification[location]

    def getOutput(self, locName, location):
        loc = self.locs[locName]
        return loc.layer_obj_heights[location]

    def getSpecialLayer(self, locName, special_layer):
        loc = self.locs[locName]
        # layer = loc.specialLayers[special_layer]
        return layer.layer_obj_heights

    def formatDataLayers(self):
        self.classifyObjHeights()
        self.normalizeAllLayers()


    # def classifyObjHeights(self):
    #     vals = []
    #     for loc in self.locs.values():
    #         vals.appent(np.squeeze(np.squeeze(loc.obj_heights)))
    #         print(vals)
    #         exit()

    def normalizeAllLayers(self):
        layer_maxs = {
                'dem':[],
                'slope':[90],
                'aspect':[359],
                'band_4':[255],
                'band_3':[255],
                'band_2':[255],
                'band_1':[255],
                'grvi':[]
                }
        layer_mins = {
                'dem':[],
                'slope':[0],
                'aspect':[0],
                'band_4':[0],
                'band_3':[0],
                'band_2':[0],
                'band_1':[0],
                'grvi':[]
                }

        for loc in self.locs.values():
            for layer_key in loc.layers.keys():
                if layer_key in layer_maxs.keys():
                    layer_maxs[layer_key].append(np.amax(loc.layers[layer_key]))
                    layer_mins[layer_key].append(np.amin(loc.layers[layer_key]))


        for loc in self.locs.values():
            for layer_key in loc.layers.keys():
                if layer_key in layer_maxs.keys():
                    layer = loc.layers[layer_key]
                    xmax, xmin = max(layer_maxs[layer_key]), min(layer_mins[layer_key])
                    layer = (layer - xmin) / (xmax - xmin)
                    loc.layers[layer_key] = layer


    def __repr__(self):
        return "Dataset({})".format(list(self.locs.values()))

class Location(object):

    def __init__(self, name, specialLayers, obj_heights=None, layers=None):
        self.name = name
        self.specialLayers = specialLayers
        self.layer_obj_heights = obj_heights if obj_heights is not None else self.loadLayerObjHeights()
        self.layers = layers if layers is not None else self.loadLayers()
        # if layers is None:
        #     self.normalizeLayers()


            # if bin_class:
            #     # self.obj_height_classification = to_categorical(self.layer_obj_heights, 2)
            #     self.obj_height_classification = self.layer_obj_heights
            # elif classify:
            #     # print("Before: ", self.layer_obj_heights)
            #     self.obj_height_classification = self.layer_obj_heights #to_categorical(self.layer_obj_heights, 4)
                # print("After: ", self.obj_height_classification)
        self.obj_height_classification = self.layer_obj_heights
        self.layerSize = list(self.layers.values())[0].shape[:2]

    def rotate(self, degrees):
        special_layers = SpecialLayer.getVegLayer(self.name)
        specialLayers = {layer_name:SpecialLayer(self.name, layer_name, degrees) for layer_name in special_layers}
        rot_layers = {}
        layer_obj_heights = self.layer_obj_heights
        layers_copy = self.layers
        for i in range(degrees//90):
            layer_obj_heights = np.rot90(layer_obj_heights)
            for j, key in enumerate(self.layers.keys()):
                layer = layers_copy[key]
                rot_layers[key] = np.rot90(layer)
        return specialLayers, layer_obj_heights, rot_layers

    def shift(self):
        special_layers = SpecialLayer.getVegLayer(self.name)
        specialLayers = {layer_name:SpecialLayer(self.name, layer_name, shift=True) for layer_name in special_layers}
        rot_layers = {}
        layer_obj_heights = self.layer_obj_heights[32:,32:]
        layers_copy = self.layers
        for j, key in enumerate(self.layers.keys()):
            layer = layers_copy[key]
            rot_layers[key] = layer[32:,32:]
        return specialLayers, layer_obj_heights, rot_layers




    def normalizeLayers(self):
        for i, key in enumerate(self.layers):
            layer = self.layers[key]
            xmax, xmin = layer.max(), layer.min()
            layer = (layer - xmin) / (xmax - xmin)
            self.layers[key] = layer



    def loadLayers(self):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if self.name in untrainged_locNames:
            directory = cwd + '/data/_untrained/{}/'.format(self.name)
        else:
            directory = cwd + '/data/{}/'.format(self.name)

        dem = np.loadtxt(directory + 'dem.txt', delimiter=',') #util.openImg(folder+'dem.tif')
        slope = np.loadtxt(directory + 'slope.txt', delimiter=',')#util.openImg(folder+'slope.tif')
        band_1 = np.loadtxt(directory + 'band_1.txt', delimiter=',')#util.openImg(folder+'band_1.tif')
        band_2 = np.loadtxt(directory + 'band_2.txt', delimiter=',')#util.openImg(folder+'band_2.tif')
        band_3 = np.loadtxt(directory + 'band_3.txt', delimiter=',')#util.openImg(folder+'band_3.tif')
        band_4 = np.loadtxt(directory + 'band_4.txt', delimiter=',')#util.openImg(folder+'band_4.tif')
        ndvi = np.loadtxt(directory + 'ndvi.txt', delimiter=',')#util.openImg(folder+'ndvi.tif')
        aspect = np.loadtxt(directory + 'aspect.txt', delimiter=',')#util.openImg(folder+'aspect.tif')
        footprints = self.loadVeg(self.name)
        print("Layers loaded for ", self.name)

        aspect[aspect>359] = 359
        aspect[aspect<0] = 0
        slope[slope>90] = 90
        slope[slope<0] = 0
        ndvi[ndvi<0] = 0

        f_32 = [dem, slope, ndvi, aspect]
        # above_zero = [dem, slope]
        u_8 = [band_1, band_2, band_3, band_4]

        for l in f_32:
            l = l.astype('float32')


        for b in u_8:
            b = b.astype('uint8')
            b[b<0] = 0
            b[b>255] = 255

        grvi = np.divide(band_4, band_2, out=np.zeros_like(band_4), where=band_2!=0)

        layers = {
                'dem':dem,
                'slope':slope,
                'aspect':aspect,
                'ndvi':ndvi,
                'band_4':band_4,
                'band_3':band_3,
                'band_2':band_2,
                'band_1':band_1,
                'footprints': footprints,
                'grvi': grvi
                }

        for val in layers.keys():
            layers[val] = layers[val][2:-2,2:-2]

        for name, layer in layers.items():
            pass
        return layers

    # @staticmethod
    def loadLayerObjHeights(self):
        cwd = os.getcwd()
        untrained_locNames = listdir_nohidden('data/_untrained/')
        if self.name in untrained_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/obj_height.txt'.format(self.name)
        else:
            fname = cwd + '/data/{}/special_layers/obj_height.txt'.format(self.name)
        obj_heights = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        obj_heights = obj_heights.astype('float32')
        obj_heights = np.around(obj_heights, 2)
        obj_heights[obj_heights > 250] = 250

        obj_heights[self.specialLayers['footprints'].allVeg == 1] = -1

        return obj_heights

    def loadVeg2(self):
        cwd = os.getcwd()
        untrained_locNames = listdir_nohidden('data/_untrained/')
        if self.name in untrained_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/footprints.txt'.format(self.name)
        else:
            fname = cwd + '/data/{}/special_layers/footprints.txt'.format(self.name)

        veg = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_COLOR)
        veg = veg.astype('uint8')

        if veg is None:
            raise RuntimeError('Could not find veg for location {} for the layer'.format(locName))
        veg = np.zeros_like(veg)
        return veg

    @staticmethod
    def loadVeg(locName, specialLayers='veg'):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if locName in untrainged_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/footprints.txt'.format(locName)
        else:
            fname = cwd + '/data/{}/special_layers/footprints.txt'.format(locName)

        veg = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_COLOR)
        veg = veg.astype('uint8')
        if veg is None:
            raise RuntimeError('Could not find veg for location {} for the layer'.format(locName))
        veg[veg!=0] = 1
        return veg

    @staticmethod
    def load(locName, specialLayers='all'):
        if specialLayers == 'all':
            special_layers = SpecialLayer.getVegLayer(locName)
        specialLayers = {layer_name:SpecialLayer(locName, layer_name) for layer_name in special_layers}

        return Location(locName, specialLayers)

    def __repr__(self):
        return "Location({}, {})".format(self.name, [d.layer_name for d in self.specialLayers.values()])

class SpecialLayer(object):

    def __init__(self, locName, layer_name, degrees=0, shift=False, allVeg=None, footprints=None, obj_heights=None):
        self.locName = locName
        self.layer_name = layer_name
        self.allVeg = allVeg if allVeg is not None else self.loadAllVeg() # 1 means not vegetation
        self.footprints = footprints     if footprints   is not None else self.loadFootprints()
        self.obj_heights = obj_heights              if obj_heights is not None else self.loadObjHeights()
        print("Special layers loaded for ", self.locName)
        if degrees > 0:
            print("before rotation: ", self.allVeg.shape)
            for i in range(degrees//90):
                self.allVeg = np.rot90(self.allVeg)
                self.footprints = np.rot90(self.footprints)
                self.obj_heights = np.rot90(self.obj_heights)
            print("after rotation: ", self.allVeg.shape)
        if shift:
            print("before cut: ", self.allVeg.shape)
            self.allVeg = self.allVeg[32:,32:]
            self.footprints = self.footprints[32:,32:]
            self.obj_heights = self.obj_heights[32:,32:]
            print("after cut: ", self.allVeg.shape)

    def loadAllVeg(self):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if self.locName in untrainged_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/footprints.txt'.format(self.locName)
        else:
            fname = cwd + '/data/{}/special_layers/footprints.txt'.format(self.locName)
        veg = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        veg = veg.astype('uint8')
        if veg is None:
            raise RuntimeError('Could not find veg for location {} for the layer {}'.format(self.locName, self.layer_name))
        veg[veg!=0] = 1
        return veg

    def loadFootprints(self):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if self.locName in untrainged_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/footprints.txt'.format(self.locName)
        else:
            fname = cwd + '/data/{}/special_layers/footprints.txt'.format(self.locName)

        not_veg = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        not_veg = not_veg.astype('uint8')
        if not_veg is None:
            raise RuntimeError('Could not open a footprint for the location {}'.format(self.locName))
        return not_veg


    def loadObjHeights(self):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if self.locName in untrainged_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/obj_height.txt'.format(self.locName)
        else:
            fname = cwd + '/data/{}/special_layers/obj_height.txt'.format(self.locName)
        obj_heights = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        obj_heights = obj_heights.astype('float32')
        obj_heights = np.around(obj_heights, 2)
        obj_heights[obj_heights > 250] = 250

        obj_heights[self.footprints == 1] = -1
        return obj_heights


    def __repr__(self):
        return "specialLayer({},{})".format(self.locName, self.layer_name)


    @staticmethod
    def getVegLayer(locName):
        vegLayers = ['footprints']
        return vegLayers

def listdir_nohidden(path):
    '''List all the files in a path that are not hidden (begin with a .)'''
    result = []

    for f in listdir(path):
        if not f.startswith('.') and not f.startswith("_"):
            result.append(f)
    return result

if __name__ == '__main__':
    raw = RawData.load()
