import math
from collections import namedtuple
import random
import json
import time
from time import localtime, strftime
from multiprocessing import Pool
from keras.utils import to_categorical


import numpy as np
# import cv2
from lib import rawdata
from lib import viz
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

AOIRadius = 11

classify = True
bin_class = False

balance_classes = False

SQUARE_DIM = 64


class Squares(object):
    '''Makes a dataset of squares for the autoencoders'''

    def __init__(self, data=None, test_set=False, mod=None, datasets=None, grab_site=False):
        print("mod: ", mod)
        self.split_beg = []
        self.split_end = []
        self.correct, self.total = {}, {}
        self.trainX, self.trainy, self.orig_trainy, self.testX, self.testy, self.orig_testy = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        if not grab_site:
            if not test_set and mod is not None:
                self.trainX, self.trainy, self.orig_trainy, self.testX, self.testy, self.orig_testy = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        else:
            if datasets is not None:
                if len(datasets) == 6:
                    self.trainX, self.trainy, self.valX, self.valy, self.testX, self.testy = datasets
                else:
                    self.squares, self.square_labels, self.square_labels_orig = datasets
                    self.makeClasses()
                    if test_set: self.trainX, self.trainy, self.orig_trainy, self.testX, self.testy, self.orig_testy = self.splitDataset()
                    else: self.trainX, self.trainy, self.square_labels_orig, self.testX, self.testy = self.squares, self.square_labels, self.square_labels_orig, [], []
                    self.makeValDataset()
            else:
                # if mod is None:
                self.data = data
                self.squares, self.square_labels, self.square_labels_orig = self.makeSquares()
                self.saveRawSquares()
                # self.measureBal()
                self.makeClasses()
                if test_set:
                    self.trainX, self.trainy, self.orig_trainy, self.testX, self.testy, self.orig_testy = self.splitDataset()
                    self.makeValDataset()
                else:
                    self.trainX, self.trainy, self.orig_trainy, self.testX, self.testy, self.orig_testy = self.squares, self.square_labels, self.square_labels_orig, self.squares, self.square_labels, self.square_labels_orig
                    if mod is None: self.makeValDataset()
                # else:
                #     self.testX, self.testy, self.square_labels_orig = [], [], []

    def setKeys(self, keys):
        for i in range(keys):
            self.correct[i] = 0
            self.total[i] = 0

    def saveRawSquares(self):
        print("Saving raw squares")
        time_string = time.strftime("%Y%m%d-%H%M%S")
        fname = "YNET_" + time_string

        np.save('output/raw_squares/' + fname + 'squares.npy', self.squares)
        np.save('output/raw_squares/' + fname + 'labels.npy', self.square_labels)
        np.save('output/raw_squares/' + fname + 'labels_orig.npy', self.square_labels_orig)

    def rotateDatasets(self, size_test=None):
        print("size test: ", size_test)
        if size_test is None: size_test = self.testX.shape[0]
        new_testX = self.trainX[:size_test]
        new_testy = self.trainy[:size_test]
        new_orig_testy = self.orig_trainy[:size_test]
        new_trainX = np.concatenate((self.trainX[size_test:], self.testX), axis=0)
        new_trainy = np.concatenate((self.trainy[size_test:], self.testy), axis=0)
        new_orig_trainy = np.concatenate((self.orig_trainy[size_test:], self.orig_testy), axis=0)
        self.trainX, self.trainy, self.orig_trainy, self.testX, self.testy, self.orig_testy = new_trainX, new_trainy, new_orig_trainy, new_testX, new_testy, new_orig_testy


    def measureBal(self):
        total = self.square_labels[0].shape[0] * self.square_labels[0].shape[1]
        print(total)
        for i, square in enumerate(self.squares):
            footprint = 0
            grass = 0
            shrub = 0
            tree = 0
            footprint = np.count_nonzero(np.argmax(self.square_labels[i], 2) == 0)
            grass = np.count_nonzero(np.argmax(self.square_labels[i], 2) == 1)
            shrub = np.count_nonzero(np.argmax(self.square_labels[i], 2) == 2)
            tree = np.count_nonzero(np.argmax(self.square_labels[i], 2) == 3)

            print(i, " - foot: ", round((footprint/total), 4)," - grass: ", round((grass/total), 4)," - shrub: ", round((shrub/total), 4)," - tree: ", round((tree/total), 4))

    def makeClasses(self):
        print(self.square_labels.shape)
        square_label = np.array(self.square_labels)
        square_label = square_label.flatten()
        sorted_squares = np.sort(square_label)
        sorted_squares = sorted_squares[sorted_squares != -1]
        if classify:
            if balance_classes:
                self.balanceClasses()
            else:
                self.setClasses()

            for i in range(len(self.split_beg)):
                print(self.split_beg[i], " to ", self.split_end[i], " len: ", np.count_nonzero((sorted_squares >= self.split_beg[i]) & (sorted_squares < self.split_end[i])))

            for i, val in enumerate(self.split_beg):
                print("greater than ", val, " and less than ", self.split_end[i], " labeled: ", i+1)
                self.square_labels[(self.square_labels >= val) & (self.square_labels < self.split_end[i])] = i+0.5
                print("count: ", np.count_nonzero(self.square_labels == (i+0.5)))

            self.square_labels+=0.5
            print("foot")
            self.square_labels[self.square_labels < 0] = 0
            print("count: ", np.count_nonzero(self.square_labels == 0))
            print("max: ", np.amax(self.square_labels))
            for i in range(5):
                print(np.count_nonzero(self.square_labels == i))
            self.square_labels = to_categorical(self.square_labels, (len(self.split_beg) + 1))


    def setClasses(self):
        ''' Set height classes manually depending on past study '''
        self.split_beg = [0,2,6,50,80]
        self.split_end = [2,6,50,80,251]

    def balanceClasses(self):
        ''' Balance height classes so that ~same amount of samples in each class '''
        for i in range(sorted_squares.shape[0]):
            try:
                split_arr = np.split(sorted_squares, 4)
                break
            except:
                sorted_squares = sorted_squares[:-1]
        previous_end = -1
        for i in split_arr:
            if i[0] <= previous_end: self.split_beg.append(previous_end)
            else: self.split_beg.append(i[0])
            if self.split_beg[-1] != i[-1]: self.split_end.append(i[-1])
            else: self.split_end.append(i[-1]+1)
            previous_end = self.split_end[-1]
        self.split_end[-1] = 251

    def makeValDataset(self):
        l = int(self.trainX.shape[0] * 0.8)
        self.valX = self.trainX[-l:]
        self.valy = self.trainy[-l:]
        self.orig_testy = self.orig_trainy[-l:]
        self.trainX = self.trainX[:l]
        self.trainy = self.trainy[:l]
        self.orig_trainy = self.orig_trainy[:l]


    def splitDataset(self):
        self.squares, self.square_labels, self.square_labels_orig = shuffle(self.squares, self.square_labels, self.square_labels_orig)
        split = 0.7
        trainX = self.squares[:int(self.squares.shape[0] * split)]
        trainy = self.square_labels[:int(self.squares.shape[0] * split)]
        orig_trainy = self.square_labels_orig[:int(self.squares.shape[0] * split)]
        testX = self.squares[int(self.squares.shape[0] * split):]
        testy = self.square_labels[int(self.squares.shape[0] * split):]
        orig_testy = self.square_labels_orig[int(self.squares.shape[0] * split):]
        return trainX, trainy, orig_trainy, testX, testy, orig_testy



# ORDER:
# dem
# slope
# aspect
# ndvi
# band_4
# band_3
# band_2
# band_1
# footprints
# grvi
    def makeSquares(self):
        all_cubes = []
        all_cubes_labels = []
        for i, loc in enumerate(self.data.locs.values()):
            print("Making squares for: ", loc.name)
            layers_arr = []
            cube = []
            for l in loc.layers.keys():
                layer = loc.layers[l]
                # print(l)
                split_indices = [SQUARE_DIM*d for d in range(1,(layer.shape[1]//SQUARE_DIM)+1)]
                h_split = np.hsplit(layer, np.array(split_indices))
                last = h_split[-1]
                if last.shape[1] < SQUARE_DIM:
                    h_split.pop()
                layer_squares = []
                for slice in h_split:
                    split_indices = [SQUARE_DIM*d for d in range(1, (layer.shape[0]//SQUARE_DIM)+1)]
                    v_split = np.vsplit(slice, np.array(split_indices))
                    last = v_split[-1]
                    if last.shape[0] < SQUARE_DIM:
                        v_split.pop()
                    layer_squares = layer_squares + v_split
                layers_arr.append(np.array(layer_squares))
            cubes = np.stack(layers_arr, axis=3)
            cube_labels = self.makeLabel(loc.layer_obj_heights)
            print(cubes.shape, " labels: ", cube_labels.shape)
            all_cubes.append(cubes)
            all_cubes_labels.append(cube_labels)
        all_cubes = np.concatenate(all_cubes, axis=0 )
        all_cubes_labels = np.concatenate(all_cubes_labels, axis=0 )
        return shuffle(all_cubes, all_cubes_labels, all_cubes_labels)

    @staticmethod
    def makeLabel(label_layer):
        squares = []
        split_indices = [SQUARE_DIM*d for d in range(1, (label_layer.shape[1]//SQUARE_DIM)+1)]
        h_split = np.hsplit(label_layer, np.array(split_indices))
        last = h_split[-1]
        if last.shape[1] < SQUARE_DIM:
            h_split.pop()
        for slice in h_split:
            split_indices = [SQUARE_DIM*d for d in range(1, (label_layer.shape[0]//SQUARE_DIM)+1)]
            v_split = np.vsplit(slice, np.array(split_indices))
            last = v_split[-1]
            if last.shape[0] < SQUARE_DIM:
                v_split.pop()
            if not classify and not bin_class:
                v_split = [np.expand_dims(v, axis=2) for v in v_split]
            squares = squares + v_split
        return np.array(squares)
