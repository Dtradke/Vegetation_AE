import numpy as np
import random
import sys
import time
import os
import pandas as pd
import tensorflow as tf


from lib import rawdata
from lib import dataset
from lib import metrics
from lib import viz
from lib import preprocess
from lib import util
from lib import model
from lib import generator
from multiprocessing import Pool
from keras.backend import manual_variable_initialization
manual_variable_initialization(True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

SPLIT = False
pretrain = False

geo_start = 0
geo_stop = 3
imagery_start = 3

def openDatasets(test_set, mod):
    data = None
    if mod is None:
        data = rawdata.RawData.load(locNames='all', special_layers='all')
        # data.formatDataLayers()
        data.normalizeAllLayers()
        masterDataSet = dataset.Squares(data, test_set, mod)
        masterDataSet.trainstring = data.names
    else:
        masterDataSet = dataset.Squares()
    if not test_set: #its the test site
        new_data = rawdata.RawData.load(locNames='untrain', special_layers='all', new_data='not_none')
        new_data.normalizeAllLayers()
        tempmasterDataSet = dataset.Squares(new_data, test_set, mod=mod, test_site=True)
        masterDataSet.teststring = new_data.names
        if mod is not None:
            masterDataSet.trainX = tempmasterDataSet.testX
            masterDataSet.trainy = tempmasterDataSet.testy
            masterDataSet.orig_trainy = tempmasterDataSet.orig_testy
        masterDataSet.testX = tempmasterDataSet.testX
        masterDataSet.testy = tempmasterDataSet.testy
        masterDataSet.orig_testy = tempmasterDataSet.orig_testy
        masterDataSet.test_ids = tempmasterDataSet.test_ids
        masterDataSet.test_arrays = tempmasterDataSet.test_arrays

    print(">>>STRINGS")
    print("train >", masterDataSet.trainstring)
    print("test  >", masterDataSet.teststring)
    return masterDataSet

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

def getModelAndTrain(masterDataSet, mod, test_set, load_datasets=False, save_mod=False):
    if mod is None:
        if SPLIT:
            X_split_1 = masterDataSet.trainX[:,:,:,:3]
            # X_split_1 = np.concatenate((np.expand_dims(masterDataSet.trainX[:,:,:,0], axis=3), (np.expand_dims(masterDataSet.trainX[:,:,:,2], axis=3))), axis=3)
            # X_split_2 = np.concatenate((np.expand_dims(masterDataSet.trainX[:,:,:,3], axis=3), masterDataSet.trainX[:,:,:,5:-1]), axis=3)
            X_split_2 = masterDataSet.trainX[:,:,:,3:-1]
            # X_split_2 = np.concatenate((masterDataSet.trainX[:,:,:,3:7],masterDataSet.trainX[:,:,:,8:-1]), axis=3)
            # X_split_2 = np.concatenate((masterDataSet.trainX[:,:,:,3:7], np.expand_dims(masterDataSet.trainX[:,:,:,8], axis=3)), axis=3)

            # X_split_1, X_split_2 = masterDataSet.trainX[:,:,:,:3], masterDataSet.trainX[:,:,:,3:-1]

            val_split_1 = masterDataSet.valX[:,:,:,:3]
            # val_split_1 = np.concatenate((np.expand_dims(masterDataSet.valX[:,:,:,0], axis=3), (np.expand_dims(masterDataSet.valX[:,:,:,2], axis=3))), axis=3)

            # val_split_2 = np.concatenate((np.expand_dims(masterDataSet.valX[:,:,:,3], axis=3), masterDataSet.valX[:,:,:,5:-1]), axis=3)
            val_split_2 = masterDataSet.valX[:,:,:,3:-1]
            # val_split_2 = np.concatenate((masterDataSet.valX[:,:,:,3:7],masterDataSet.valX[:,:,:,8:-1]), axis=3)
            # val_split_2 = np.concatenate((masterDataSet.valX[:,:,:,3:7], np.expand_dims(masterDataSet.valX[:,:,:,8], axis=3)), axis=3)



            # val_split_1, val_split_2 = masterDataSet.valX[:,:,:,:3], masterDataSet.valX[:,:,:,3:-1]
            print("Split shape: ", X_split_1.shape, " ", X_split_2.shape)
            print("Val Split shape: ", val_split_1.shape, " ", val_split_2.shape)
            inputs = [X_split_1, X_split_2]
            vals = [val_split_1, val_split_2]
            if pretrain:
                pretrain_mod = model.unet_split(X_split_1, X_split_2, pretrain=True)
                mod = model.pretrainYNET(inputs, vals, masterDataSet, pretrain_mod, mod)
            else:
                mod = model.ynet_split(X_split_1, X_split_2)
                # mod = model.unet_branch_dropout(X_split_1, X_split_2)

            # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            # gen = generator.DataGenerator(masterDataSet.trainX, masterDataSet.trainy)
            # val_gen = generator.DataGenerator(masterDataSet.valX, masterDataSet.valy)
            # mod.fit(gen, epochs=300, validation_data=val_gen, callbacks=[es])


# TODO: do transfer learning with small datasets after unsupervised pretraining... see how small the dataset can be
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            # mc = ModelCheckpoint('models/split_nodrop_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True)
            mod.fit( inputs, masterDataSet.trainy, batch_size=32, epochs=300, verbose=1, validation_data=(vals, masterDataSet.valy), callbacks=[es]) #, callbacks=[es, mc]
        else:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            mod = model.unet(masterDataSet)
            mod.fit(masterDataSet.trainX, masterDataSet.trainy, batch_size=32, epochs=300, verbose=1, validation_data=(masterDataSet.valX, masterDataSet.valy), callbacks=[es])

        if save_mod: util.saveExperiment(mod, masterDataSet, test_set, SPLIT)
    else:
        if SPLIT: mod = model.ynet_split(masterDataSet.trainX[:,:,:,:3], masterDataSet.trainX[:,:,:,3:], pretrained_weights=mod)
        else: mod = model.unet(masterDataSet, pretrained_weights=mod)
    return mod

def modPredict(mod, masterDataSet):
    print("Predicting...")
    if SPLIT:
        # X_split_1, X_split_2 = masterDataSet.testX[:,:,:,:3], masterDataSet.testX[:,:,:,3:-1]
        X_split_1 = masterDataSet.testX[:,:,:,:3]
        # X_split_1 = np.concatenate((np.expand_dims(masterDataSet.testX[:,:,:,0], axis=3), (np.expand_dims(masterDataSet.testX[:,:,:,2], axis=3))), axis=3)

        # X_split_1 = np.column_stack((masterDataSet.testX[:,:,:,0], masterDataSet.testX[:,:,:,2]), axis=3) #np.stack((masterDataSet.trainX[:,:,:,3],masterDataSet.trainX[:,:,:,5:]), axis=3)#
        X_split_2 = masterDataSet.testX[:,:,:,3:-1]
        # X_split_2 = np.concatenate((masterDataSet.testX[:,:,:,3:7],masterDataSet.testX[:,:,:,8:-1]), axis=3)
        # X_split_2 = np.concatenate((np.expand_dims(masterDataSet.testX[:,:,:,3], axis=3), masterDataSet.testX[:,:,:,5:-1]), axis=3)
        y_preds = mod.predict([X_split_1, X_split_2])
    else:
        y_preds = mod.predict(masterDataSet.testX)

    # np.save("ynet_squares_ground.npy", masterDataSet.testy)
    # np.save("ynet_squares_pred.npy", y_preds)
    util.evaluateYNET(y_preds, masterDataSet)

def heightsCheck(masterDataSet):
    total_val = {}
    for i in range(len(masterDataSet.split_beg)+1):
        total_val[i] = 0
    for vall in masterDataSet.testy:
        pred, val = util.formatPreds(vall, vall)
        total_val[0]+=np.count_nonzero(val == 0)
        total_val[1]+=np.count_nonzero(val == 1)
        total_val[2]+=np.count_nonzero(val == 2)
        total_val[3]+=np.count_nonzero(val == 3)
        total_val[4]+=np.count_nonzero(val == 4)
        total_val[5]+=np.count_nonzero(val == 5)
        total_val[6]+=np.count_nonzero(val == 6)
    [print(total_val[i]) for i in total_val.keys()]

def regHeightsCheck(masterDataSet, mod):
    flat_train = masterDataSet.trainy.flatten()
    flat_val = masterDataSet.valy.flatten()

    # imperial
    # bottom = [0,2,6,6,20,50,80]
    # top = [2,6,20,50,50,80,251]
    # metric
    bottom = [0,0.6,1.83,1.83,6,15.25,24.4]
    top = [0.6,1.83,6,15.25,15.25,24.4,77]
    for i, lower in enumerate(bottom):
        train = flat_train[(flat_train >= lower) & (flat_train < top[i])]
        val = flat_val[(flat_val >= lower) & (flat_val < top[i])]
        print("Height ", lower, " to ", top[i], " - train: ", train.size, " - val: ", val.size)

def openAndTrain(test_set=True, mod=None, load_datasets=None, save_mod=False):
    start_time = time.time()
    if load_datasets is not None:
        try:
            print("Loading preprocessed datasets")
            datasets = util.loadDatasets(load_datasets, save_mod)
        except:
            print("Loading Squares")
            datasets = util.loadSquareDatasets(load_datasets)
        masterDataSet = dataset.Squares(test_set=test_set, datasets=datasets)
    else:
        print("Making new Datasets")
        masterDataSet = openDatasets(test_set, mod)

    try:
        heightsCheck(masterDataSet)
        test_len = util.KCross(masterDataSet)
    except:
        if test_set == True:
            regHeightsCheck(masterDataSet)
        test_len = 1

    for i in range(test_len):
        print(i)
        print("Length of train: ", masterDataSet.trainX.shape[0], " and test: ", masterDataSet.testX.shape[0])
        if test_set and not load_datasets: mod=None
        mod = getModelAndTrain(masterDataSet, mod, test_set, load_datasets, save_mod)
        # print(mod.get_weights())
        modPredict(mod, masterDataSet)

        # NOTE: NOT K-CROSS VALIDATION
        break

        if remainder != 0 and i == (test_len - 2):
            print("remainder and next test: ", remainder)
            masterDataSet.rotateDatasets(size_test=remainder)
        else:
            masterDataSet.rotateDatasets()
    viz.displayKCrossVal(masterDataSet)


if __name__ == "__main__":
    if 'test_set' in sys.argv:
        print("========= TEST SET =========")
        if len(sys.argv) > 2:
            if sys.argv[-1] == 'train': #python3 autoencoder.py test_set [dataset string] train
                print("loading datasets but training new model")
                openAndTrain(True, load_datasets=sys.argv[-2], save_mod=True)
            elif len(sys.argv) == 4: #python3 autoencoder.py test_set [model string] [dataset_string]
                print("Loading past model and other datasets")
                openAndTrain(True, mod=sys.argv[-2], load_datasets=sys.argv[-1])
            else: #python3 autoencoder.py test_set [model string]
                print("Loading datasets and model with same name")
                openAndTrain(True, mod=sys.argv[-1], load_datasets=sys.argv[-1])
        else: #python3 autoencoder.py test_set
            print("Loading new datasets and training new model")
            openAndTrain(True, save_mod=True)
    else: #python3 autoencoder.py
        print("========= TEST SITE =========")
        if sys.argv[-1] == 'train':
            print("Loading datasets and training new model: ", sys.argv[-1])
            openAndTrain(False, load_datasets=sys.argv[-1], save_mod=True)
        if len(sys.argv) == 2:
            print("Loading model: ", sys.argv[-1])
            openAndTrain(False, mod=sys.argv[-1])
            # openAndTrain(False, mod=sys.argv[-1], load_datasets=sys.argv[-1])
        else:
            openAndTrain(False, save_mod=True)
