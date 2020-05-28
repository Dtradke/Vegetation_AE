import numpy as np
import random
import sys
import time
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, X_data, y_data,
                 to_fit=True, batch_size=32, dim=(64, 64),
                 n_channels=10, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = range(X_data.shape[0])#list_IDs
        # self.labels = labels
        # self.image_path = image_path
        # self.mask_path = mask_path
        self.X_data = X_data
        self.y_data = y_data
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(self.X_data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.X_data #self._generate_X(list_IDs_temp)
        y = self.y_data

        x_batch_1 = np.empty(((self.batch_size*4), *self.dim, 3))
        x_batch_2 = np.empty(((self.batch_size*4), *self.dim, 7))
        y_batch = np.empty(((self.batch_size*4), *self.dim,1), dtype=int)

        count = 0
        for i, val in enumerate(indexes):
            x_batch_1[count,] = X[val][:,:,:3]
            x_batch_2[count,] = X[val][:,:,3:]
            y_batch[count,] = y[val]
            count+=1
            rot = 1
            while rot < 4:
                # print("rot: ", rot)
                x_batch_1[count,] = np.rot90(X[val][:,:,:3], rot, (1,0))
                x_batch_2[count,] = np.rot90(X[val][:,:,3:], rot, (1,0))
                y_batch[count,] = np.rot90(y[val], rot)

                rot+=1
                count+=1

        # print("np.array(x_batch_1)> ", x_batch_1.shape)
        # print("np.array(x_batch_2)> ", x_batch_2.shape)
        x_batch_1, x_batch_2, y_batch = shuffle(x_batch_1, x_batch_2, y_batch)

        return [x_batch_1, x_batch_2], y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        # self.indexes = np.arange(len(self.list_IDs))
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
