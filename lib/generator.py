import numpy as np
import random
import sys
import time
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence


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
        return int(np.floor((self.X_data.shape[0] * 4) / self.batch_size))

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

        x_batch_1, x_batch_2 = [], []
        y_batch = []
        for i, val in enumerate(indexes):
            # print(">X: ", X[val][:,:,:3].shape)
            # print(">X2: ", X[val][:,:,3:].shape)
            # print("y: ", y[val].shape)

            x_batch_1.append(X[val][:,:,:3])
            x_batch_2.append(X[val][:,:,3:])
            print("x: ", len(x_batch_1))
            y_batch.append(y[val])
            rot = 1
            while rot < 4:
                x_batch_1.append(np.rot90(X[val][:,:,:3], rot, (1,2)))
                x_batch_2.append(np.rot90(X[val][:,:,3:], rot, (1,2)))
                y_batch.append(np.rot90(y[val]), rot)
                rot+=1

        # print(">X - : ", np.array(x_batch_1).shape)
        # print(">y - : ", y.shape)

        print("np.array(x_batch_1)> ", np.array(x_batch_1).shape)
        print("np.array(x_batch_2)> ", np.array(x_batch_2).shape)

        return [np.array(x_batch_1), np.array(x_batch_2)], np.array(y_batch)

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        # self.indexes = np.arange(len(self.list_IDs))
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_grayscale_image(self.image_path + self.labels[ID])

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim,1), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self._load_grayscale_image(self.mask_path + self.labels[ID])

        return y

    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        return img
