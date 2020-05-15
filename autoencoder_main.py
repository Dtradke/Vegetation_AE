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
from multiprocessing import Pool
from keras.backend import manual_variable_initialization
manual_variable_initialization(True)

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

SPLIT = True
pretrain = True

def openDatasets(test_set, mod):
    data = []
    if mod is None:
        data = rawdata.RawData.load(locNames='all', special_layers='all')
    # data.formatDataLayers()
    data.normalizeAllLayers()
    masterDataSet = dataset.Squares(data, test_set, mod)
    if not test_set: #its the test site
        new_data = rawdata.RawData.load(locNames='untrain', special_layers='all', new_data='not_none')
        testSiteDataset = dataset.Squares(new_data, test_set, mod=None)
        masterDataSet.testX = testSiteDataset.squares
        masterDataSet.testy = testSiteDataset.square_labels
        masterDataSet.square_labels_orig = testSiteDataset.square_labels_orig
    return masterDataSet

def getModelAndTrain(masterDataSet, mod, test_set, load_datasets=False):
    if mod is None:
        if SPLIT:
            X_split_1, X_split_2 = masterDataSet.trainX[:,:,:,:3], masterDataSet.trainX[:,:,:,3:]
            val_split_1, val_split_2 = masterDataSet.valX[:,:,:,:3], masterDataSet.valX[:,:,:,3:]
            print("Split shape: ", X_split_1.shape, " ", X_split_2.shape)
            inputs = [X_split_1, X_split_2]
            vals = [val_split_1, val_split_2]
            if pretrain:
                pretrain_mod = model.unet_split(X_split_1, X_split_2, pretrain=True)
                mod = model.pretrainYNET(inputs, vals, masterDataSet, pretrain_mod, mod)

# TODO: do transfer learning with small datasets after unsupervised pretraining... see how small the dataset can be

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            # mc = ModelCheckpoint('models/split_nodrop_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True)
            mod.fit( inputs, masterDataSet.trainy, batch_size=32, epochs=100, verbose=1, validation_data=(vals, masterDataSet.valy), callbacks=[es]) #, callbacks=[es, mc]
        else:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            mod = model.unet(masterDataSet)
            mod.fit(masterDataSet.trainX, masterDataSet.trainy, batch_size=32, epochs=1, verbose=1, validation_data=(masterDataSet.valX, masterDataSet.valy), callbacks=[es])
        if not load_datasets: util.saveExperiment(mod, masterDataSet, test_set, SPLIT)
    else:
        if SPLIT: mod = model.unet_split(masterDataSet.trainX[:,:,:,:3], masterDataSet.trainX[:,:,:,3:], pretrained_weights=sys.argv[-1])
        else: mod = model.unet(masterDataSet, pretrained_weights='models/' + sys.argv[2])
    return mod

def modPredict(mod, masterDataSet):
    print("Predicting...")
    if SPLIT:
        X_split_1, X_split_2 = masterDataSet.testX[:,:,:,:3], masterDataSet.testX[:,:,:,3:]
        y_preds = mod.predict([X_split_1, X_split_2])
    else:
        y_preds = mod.predict(masterDataSet.testX)
    util.evaluateYNET(y_preds, masterDataSet)

def openAndTrain(test_set=True, mod=None, load_datasets=False):
    start_time = time.time()
    if load_datasets:
        try:
            print("Loading preprocessed datasets")
            datasets = util.loadDatasets(mod)
        except:
            print("Loading Squares")
            datasets = util.loadSquareDatasets(mod)
        masterDataSet = dataset.Squares(test_set=test_set, datasets=datasets)
    else:
        masterDataSet = openDatasets(test_set, mod)

    total_val = {}
    for i in range(5):
        total_val[i] = 0
    for val in masterDataSet.testy:
        pred, val = util.formatPreds(val, val)
        total_val[0]+=np.count_nonzero(val == 0)
        total_val[1]+=np.count_nonzero(val == 1)
        total_val[2]+=np.count_nonzero(val == 2)
        total_val[3]+=np.count_nonzero(val == 3)
        total_val[4]+=np.count_nonzero(val == 4)
    print(pred)
    [print(total_val[i]) for i in total_val.keys()]
    exit()

    test_len = util.KCross(masterDataSet)

    for i in range(test_len):
        print(i)
        print("Length of train: ", masterDataSet.trainX.shape[0], " and test: ", masterDataSet.testX.shape[0])
        if test_set and not load_datasets: mod=None
        mod = getModelAndTrain(masterDataSet, mod, test_set, load_datasets)
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
            if sys.argv[-1] == 'train': #python3 autoencoder.py test_set [model string] train
                print("loading datasets but training new model")
                openAndTrain(True, load_datasets=True)
            else: #python3 autoencoder.py test_set [model string]
                print("Loading datasets and model")
                openAndTrain(True, mod=sys.argv[2], load_datasets=True)
        else: #python3 autoencoder.py test_set
            print("Loading new datasets and training new model")
            openAndTrain(True)
    else: #python3 autoencoder.py
        print("========= TEST SITE =========")
        if 'mod' in sys.argv: openAndTrain(False, mod='mod')
        else: openAndTrain(False)
