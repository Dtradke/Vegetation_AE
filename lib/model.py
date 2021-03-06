from time import localtime, strftime
import os
import time
import tensorflow as tf
from keras import backend as K
import sys


import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import optimizers
# from keras import backend as keras

from keras.layers import Lambda
from keras.layers import add

import keras.models
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate, Input
from keras.layers import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.losses import mse, binary_crossentropy

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from numpy.random import seed
seed(42)# keras seed fixing
tf.random.set_seed(42)# tensorflow seed fixing

import random

classify = False
bin_class = False

GPU = False


try:
    from lib import preprocess
except:
    import preprocess


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def vae(masterDataSet, pretrained_weights = None):
    original_dim = masterDataSet.testX[0].shape[0] * masterDataSet.testX[0].shape[0]
    input_shape = (original_dim, )
    intermediate_dim = 512
    batch_size = 128
    latent_dim = 2

    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    return vae

def unet(masterDataSet, pretrained_weights = None):
    if pretrained_weights:
        fname = 'models/' + pretrained_weights + ".h5"
        model = load_model(fname)
        return model

    input_size = masterDataSet.testX[0].shape

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    if classify:
        conv9 = Conv2D(12, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) #putting these in here better for trees
        conv10 = Conv2D(6, 1, activation = 'softmax')(conv9)
    elif bin_class:
        conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)
    else:
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'linear')(conv9)


    model = Model(input = inputs, output = conv10)

    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy'])
    # NOTE: think about mse loss because of softmax
    # model.summary()

    # if(pretrained_weights):
    # 	model.load_weights(pretrained_weights)

    return model

def encoder(inputs):
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

    return conv1, conv2, conv3, drop4, conv5


def ynet_split(X_split_1, X_split_2, pretrain=False, pretrained_weights = None):
    if pretrained_weights:
        fname = 'models/' + pretrained_weights + ".h5"
        model = load_model(fname)
        return model

    input_size_1 = X_split_1[0].shape
    input_size_2 = X_split_2[0].shape

    inputs_1 = Input(input_size_1)
    inputs_2 = Input(input_size_2)

    conv1_1, conv2_1, conv3_1, drop4_1, conv5_1 = encoder(inputs_1)
    conv1_2, conv2_2, conv3_2, drop4_2, conv5_2 = encoder(inputs_2)

    return_merge = concatenate([conv5_1,conv5_2], axis = 3)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(return_merge)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4_1,up6], axis = 3)
    merge6 = concatenate([drop4_2,merge6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3_2,up7], axis = 3)
    merge7 = concatenate([conv3_1,merge7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2_1,up8], axis = 3)
    merge8 = concatenate([conv2_2,merge8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1_2,up9], axis = 3)
    merge9 = concatenate([conv1_1,merge9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    if pretrain:
        conv9 = Conv2D(20, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(10, 1, activation = 'linear')(conv9)
    elif classify:
            conv9 = Conv2D(10, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
            conv10 = Conv2D(5, 1, activation = 'softmax')(conv9)
    elif bin_class:
        conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)
    else:
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

    model = Model(input = [inputs_1, inputs_2], output = conv10)


    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy']) #mse

    # model.summary()



    return model

def pretrainYNET(inputs, vals, masterDataSet, pretrain_mod, mod):
    pretrain_mod.fit(inputs, masterDataSet.trainX, batch_size=32, epochs=30, verbose=1, validation_data=(vals, masterDataSet.valX))
    # for layer in pretrain_mod.layers[:-2]:
    #     layer.trainable = False

    pretrain_mod.layers.pop()
    pretrain_mod.layers.pop()
    conv9 = pretrain_mod.layers[-2].output
    conv9 = Conv2D((masterDataSet.trainy.shape[-1]*2), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(masterDataSet.trainy.shape[-1], 1, activation = 'softmax')(conv9)
    mod = Model(pretrain_mod.input, conv10)
    mod.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy']) #mse
    return mod

def unet_branch_dropout(X_split_1, X_split_2, pretrained_weights = None):
    input_size_1 = X_split_1[0].shape
    input_size_2 = X_split_2[0].shape

    inputs_1 = Input(input_size_1)
    inputs_2 = Input(input_size_2)

    conv1_1, conv2_1, conv3_1, drop4_1, conv5_1 = encoder(inputs_1)
    conv1_2, conv2_2, conv3_2, drop4_2, conv5_2 = encoder(inputs_2)

    return_merge = concatenate([conv5_1,conv5_2], axis = 3)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(return_merge)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    conc_4 = concatenate([drop4_1, drop4_2], axis=3)
    conc_4 = Reshape((2, 8, 8, 512))(conc_4)
    dropout_layer = Dropout(rate=0.5, noise_shape=[None, 2, 1, 1, 1])(conc_4)
    dropout_layer_0, dropout_layer_1 = keras.layers.Lambda(lambda tensor: tf.split(tensor, 2, axis=1))(dropout_layer)
    dropout_layer_0 = keras.layers.Lambda(lambda tensor: tf.squeeze(tensor, 1))(dropout_layer_0)
    dropout_layer_1 = keras.layers.Lambda(lambda tensor: tf.squeeze(tensor, 1))(dropout_layer_1)
    conc_4 = concatenate([dropout_layer_0, up6])
    #more decoding layers
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conc_4)
    conv6 = concatenate([dropout_layer_1, conv6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    conc_3 = concatenate([conv3_1, conv3_2], axis=3)
    conc_3 = Reshape((2, 16, 16, 256))(conc_3)
    dropout_layer = Dropout(rate=0.5, noise_shape=[None, 2, 1, 1, 1])(conc_3)
    dropout_layer_0, dropout_layer_1 = keras.layers.Lambda(lambda tensor: tf.split(tensor, 2, axis=1))(dropout_layer)
    dropout_layer_0 = keras.layers.Lambda(lambda tensor: tf.squeeze(tensor, 1))(dropout_layer_0)
    dropout_layer_1 = keras.layers.Lambda(lambda tensor: tf.squeeze(tensor, 1))(dropout_layer_1)
    conc_3 = concatenate([dropout_layer_1, up7])
    #more decoding layers
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conc_3)
    conc_3 = concatenate([dropout_layer_0, conv7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conc_3)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    conc_2 = concatenate([conv2_1, conv2_1], axis=3)
    conc_2 = Reshape((2, 32, 32, 128))(conc_2)
    dropout_layer = Dropout(rate=0.5, noise_shape=[None, 2, 1, 1, 1])(conc_2)
    dropout_layer_0, dropout_layer_1 = keras.layers.Lambda(lambda tensor: tf.split(tensor, 2, axis=1))(dropout_layer)
    dropout_layer_0 = keras.layers.Lambda(lambda tensor: tf.squeeze(tensor, 1))(dropout_layer_0)
    dropout_layer_1 = keras.layers.Lambda(lambda tensor: tf.squeeze(tensor, 1))(dropout_layer_1)
    conc_2 = concatenate([dropout_layer_0, up8])
    #more decoding layers
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conc_2)
    conc_2 = concatenate([dropout_layer_1, conv8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conc_2)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    conc_1 = concatenate([conv1_1, conv1_2], axis=3)
    conc_1 = Reshape((2, 64, 64, 64))(conc_1)
    dropout_layer = Dropout(rate=0.5, noise_shape=[None, 2, 1, 1, 1])(conc_1)
    dropout_layer_0, dropout_layer_1 = keras.layers.Lambda(lambda tensor: tf.split(tensor, 2, axis=1))(dropout_layer)
    dropout_layer_0 = keras.layers.Lambda(lambda tensor: tf.squeeze(tensor, 1))(dropout_layer_0)
    dropout_layer_1 = keras.layers.Lambda(lambda tensor: tf.squeeze(tensor, 1))(dropout_layer_1)
    conc_1 = concatenate([dropout_layer_1, up9])
    #more decoding layers
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conc_1)
    conc_1 = concatenate([dropout_layer_0, conv9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    if classify:
        conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(4, 1, activation = 'softmax')(conv9)
    elif bin_class:
        conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)
    else:
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

    model = Model(input = [inputs_1, inputs_2], output = conv10)



    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy'])

    # model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

# NOTE: REGULARIZE BY RANDOM CONCAT OF THESE INSTEAD
def euclidean_distance(A, B):
    rshpA = K.expand_dims( A,axis = 0) #1
    rshpB = K.expand_dims( B,axis = 0)
    diff= rshpA-rshpB
    return diff

# Lambda for subtracting two tensors
def getMSE(r1, r2):
    # minus_r2 = Lambda(lambda x: -x)(r2)
    # subtracted = add([r1,minus_r2])
    # out= Lambda(lambda x: x**2)(subtracted)
    # model = Model([r1,r2],out)
    return K.mean(K.square(K.stack(r1) - K.stack(r2)))
    # return model

class BaseModel(object):

    DEFAULT_EPOCHS = 1
    DEFAULT_BATCHSIZE = 1000

    def __init__(self, kerasModel=None, preProcessor=None):
        self.kerasModel = kerasModel
        self.preProcessor = preProcessor

    def fit(self, trainingDataset, validatateDataset=None, epochs=DEFAULT_EPOCHS,batch_size=DEFAULT_BATCHSIZE):
        assert self.kerasModel is not None, "You must set the kerasModel within a subclass"
        assert self.preProcessor is not None, "You must set the preProcessor within a subclass"

        print('training on ', trainingDataset)
        # get the actual samples from the collection of points
        (tinputs, toutputs), ptList = self.preProcessor.process(trainingDataset)
        if validatateDataset is not None:
            (vinputs, voutputs), ptList = self.preProcessor.process(validatateDataset)
            history = self.kerasModel.fit(tinputs, toutputs, batch_size=batch_size, epochs=epochs, validation_data=(vinputs, voutputs))
        else:
            history = self.kerasModel.fit(tinputs, toutputs, batch_size=batch_size, epochs=epochs)
        return history

    def predict(self, dataset):
        assert self.kerasModel is not None, "You must set the kerasModel within a subclass"
        print("In predict")
        (inputs, outputs), ptList = self.preProcessor.process(dataset)
        results = self.kerasModel.predict(inputs).flatten()
        if mode:
            resultDict = {pt:random.utility(0.0, 1.0) for (pt, pred) in zip(ptList, results)}
        else:
            resultDict = {pt:pred for (pt, pred) in zip(ptList, results)}
        return resultDict

    def save(self, name=None):
        if name is None:
            name = strftime("%d%b%H_%M", localtime())
        if "models/" not in name:
            name = "models/" + name
        if not name.endswith('/'):
            name += '/'

        if not os.path.isdir(name):
            os.mkdir(name)

        className = str(self.__class__.__name__)
        with open(name+'class.txt', 'w') as f:
            f.write(className)
        self.kerasModel.save(name+'model.h5')

def load(modelFolder):
    if 'models/' not in modelFolder:
        modelFolder = 'models/' + modelFolder
    assert os.path.isdir(modelFolder), "{} is not a folder".format(modelFolder)

    if not modelFolder.endswith('/'):
        modelFolder += '/'

    modelFile = modelFolder + 'model.h5'
    model = keras.models.load_model(modelFile)

    objFile = modelFolder + 'class.txt'
    with open(objFile, 'r') as f:
        classString = f.read().strip()
    class_ = globals()[classString]
    obj = class_(kerasModel=model)

    return obj


class HeightModel(Sequential): #Model

    def __init__(self, preProcessor, weightsFileName=None):
        self.preProcessor = preProcessor

        kernelDiam = 2*self.preProcessor.AOIRadius+1

        super().__init__()
        # there is also the starting perim which is implicitly gonna be included
        nchannels = len(self.preProcessor.whichLayers)
        nchannels += 1
        input_shape = (kernelDiam, kernelDiam, nchannels)
        print(input_shape)

        # class project model
        self.add(Conv2D(32, kernel_size=(3,3), strides=(1,1),activation='relu', input_shape=input_shape)) #, input_shape=input_shape
        self.add(Conv2D(32, (3,3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.add(Dropout(0.3))

        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.add(Dropout(0.3))

        self.add(Flatten())
        self.add(Dense(96, kernel_initializer = 'normal', activation = 'relu',name='first_dense'))
        self.add(Dropout(0.3))
        self.add(Dense(160, kernel_initializer = 'normal', activation = 'relu',name='output'))

        #play model
        # self.add(Conv2D(32, kernel_size=(2,2), strides=(1,1),activation='relu', input_shape=input_shape)) #, input_shape=input_shape
        # self.add(Conv2D(64, (2,2), activation='relu'))
        # self.add(MaxPooling2D(pool_size=(2,2))) #, strides=(2,2)
        # self.add(Dropout(0.3))
        #
        # self.add(Conv2D(128, (2,2), activation='relu'))
        # self.add(Conv2D(256, (2,2), activation='relu'))
        # self.add(MaxPooling2D(pool_size=(2,2)))
        # # self.add(Dropout(0.3))
        #
        # self.add(Flatten())
        # self.add(Dense(128, kernel_initializer = 'normal', activation = 'relu',name='first_dense'))
        # self.add(Dropout(0.3))
        # self.add(Dense(64, kernel_initializer = 'normal', activation = 'relu',name='output'))

        opt = SGD(lr=0.01)
        if classify:
            if bin_class:
                print("IN BIN CLASS OUTPUT")
                self.add(Dense(2, activation='softmax', name='bin_final_output'))

                print(self.summary())
                self.compile(optimizer='rmsprop', #'rmsprop'
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
            else:
                print("IN CLASS OUTPUT")
                self.add(Dense(4, activation='softmax', name='4_class_final_output'))
                print(self.summary())
                self.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        else:
            print("IN REGRESS OUTPUT")
            self.add(Dense(1, kernel_initializer = 'normal', activation = 'relu',name='regress_final_output'))
            print(self.summary())

            # self.compile(optimizer=opt, loss='mean_squared_error')
            self.compile(optimizer='adam', loss='mean_squared_error')

        if weightsFileName is not None:
            self.load_weights(weightsFileName)

    def fit(self, training, validate, pp, epochs=1):
        if GPU:
            trainPtList = training.toList(training.points)
            valPtList = validate.toList(validate.points)
            partition = {'train':[None] * len(trainPtList),
                        'validation':[None] * len(valPtList)}
            labels = {}
            t_idx = 0
            for pt in trainPtList:
                # locName, location = pt
                partition['train'][t_idx] = pt
                labels[pt] = t_idx
                t_idx += 1

            v_idx = 0
            for pt in valPtList:
                partition['validation'][v_idx] = pt
                labels[pt] = v_idx
                v_idx += 1

            aoi_size = (2 * pp.AOIRadius) + 1

            params = {'dim': (aoi_size, aoi_size),
              'batch_size': 5,
              'n_channels': len(pp.whichLayers) + 1,
              'shuffle': True,
              'whichLayers': pp.whichLayers,
              'AOIRadius': pp.AOIRadius,
              'dataset': training}

            # print("partition: ", partition)
            # print('labels: ', labels)


            training_generator = preprocess.DataGenerator(partition['train'], labels, **params)
            validation_generator = preprocess.DataGenerator(partition['validation'], labels, **params)


            history = super().fit_generator(generator=training_generator, epochs=epochs, validation_data=validation_generator, workers=0)
        else:
            print("JUST FIT")
            # get the actual samples from the collection of points
            (tinputs, toutputs), ptList = self.preProcessor.process(training)
            (vinputs, voutputs), ptList = self.preProcessor.process(validate)
            print('training on ', training)

            if classify:
                history = super().fit(tinputs, toutputs, batch_size=100000, epochs=epochs, validation_data=(vinputs, voutputs))
            else:
                # history = super().fit(tinputs, toutputs, batch_size=100000, epochs=epochs, validation_data=(vinputs, voutputs))
                history = super().fit(tinputs, toutputs, batch_size=32, epochs=epochs, validation_data=(vinputs, voutputs))

        # temp = self.saveWeights()
        return history

    def saveWeights(self, fname=None):
        if fname is None:
            timeString = time.strftime("%m%d-%H%M%S")
            fname = 'models/{}_'.format(timeString)
            if classify:
                if bin_class:
                    fname = fname + "classifyBIN"
                else:
                    fname = fname + "classify"
            else:
                fname = fname + "regress"

        return fname
        # self.save_weights(fname)
        # return fname

    def predict(self, dataset, mode):
        print('start predict')
        (inputs, outputs), ptList = self.preProcessor.process(dataset)

        results = super().predict(inputs).flatten()

        if classify:
            if bin_class:
                big_results = []
                little_arr = []
                count = 0
                assert len(results)%2 == 0
                for i in results:
                    if count == 2:
                        count = 0
                        assert round(sum(little_arr), 3) == 1.0
                        big_results.append(little_arr)
                        little_arr = []

                    little_arr.append(i)
                    count = count + 1

                results = big_results
            else:
                big_results = []
                little_arr = []
                count = 0
                assert len(results)%4 == 0
                for i in results:
                    if count == 4:
                        count = 0
                        assert round(sum(little_arr), 3) == 1.0
                        big_results.append(little_arr)
                        little_arr = []

                    little_arr.append(i)
                    count = count + 1

                results = big_results

        if mode:
            resultDict = {pt:random.uniform(0.0,1.0) for (pt, pred) in zip(ptList, results)}
        else:
            resultDict = {pt:pred for (pt, pred) in zip(ptList, results)}
        print("end predict")
        return resultDict, results

class OurModel(BaseModel):

    def __init__(self, kerasModel=None):
        usedLayers = ['dem','ndvi', 'aspect', 'band_1', 'band_2', 'band_3', 'band_4', 'slope', 'grvi']
        AOIRadius = 5
        pp = preprocess.PreProcessor(usedLayers, AOIRadius)

        if kerasModel is None:
            kerasModel = self.createModel(pp)

        super().__init__(kerasModel, pp)

    @staticmethod
    def createModel(pp):
        # make our keras Model
        kernelDiam = 2*pp.AOIRadius+1
        ib = ImageBranch(len(pp.whichLayers), kernelDiam)

        kerasModel = ImageBranch(len(pp.whichLayers), kernelDiam)
        return kerasModel

class OurModel2(BaseModel):
    pass

if __name__ == '__main__':
    m = OurModel()
    m.save()

    n = load('models/15Nov09_41')
    print(n)
