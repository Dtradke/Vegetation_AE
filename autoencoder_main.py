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

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

SPLIT = True

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

def getModelAndTrain(masterDataSet, mod, test_set):
    if mod is None:
        if SPLIT:
            X_split_1, X_split_2 = masterDataSet.trainX[:,:,:,:3], masterDataSet.trainX[:,:,:,3:]
            val_split_1, val_split_2 = masterDataSet.valX[:,:,:,:3], masterDataSet.valX[:,:,:,3:]
            print("Split shape: ", X_split_1.shape, " ", X_split_2.shape)
            mod = model.unet_split(X_split_1, X_split_2)
            inputs = [X_split_1, X_split_2]
            vals = [val_split_1, val_split_2]

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            # mc = ModelCheckpoint('models/split_nodrop_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True)
            mod.fit( inputs, masterDataSet.trainy, batch_size=32, epochs=1, verbose=1, validation_data=(vals, masterDataSet.valy), callbacks=[es]) #, callbacks=[es, mc]
        else:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            mod = model.unet(masterDataSet)
            mod.fit(masterDataSet.trainX, masterDataSet.trainy, batch_size=32, epochs=300, verbose=1, validation_data=(masterDataSet.valX, masterDataSet.valy), callbacks=[es])
        # util.saveExperiment(mod, masterDataSet, test_set, SPLIT)
    else:
        if SPLIT: mod = model.unet_split(masterDataSet, pretrained_weights='models/20200421-015819_UNET-test_site.h5')
        else: mod = model.unet(masterDataSet, pretrained_weights='models/20200421-015819_UNET-test_site.h5')
    return mod

def modPredict(mod, masterDataSet):
    if SPLIT:
        X_split_1, X_split_2 = masterDataSet.testX[:,:,:,:3], masterDataSet.testX[:,:,:,3:]
        y_preds = mod.predict([X_split_1, X_split_2])
    else:
        y_preds = mod.predict(masterDataSet.testX)
    util.evaluateYNET(y_preds, masterDataSet)

def openAndTrain(test_set=True, mod=None, load_datasets=False):
    start_time = time.time()
    if load_datasets:
        datasets = util.loadDatasets()
        masterDataSet = dataset.Squares(datasets=datasets)
    else:
        masterDataSet = openDatasets(test_set, mod)
    test_len = (masterDataSet.trainX.shape[0] // masterDataSet.testX.shape[0])+1
    if (masterDataSet.trainX.shape[0] % masterDataSet.testX.shape[0]) != 0:
        remainder = (masterDataSet.trainX.shape[0] % masterDataSet.testX.shape[0])
        test_len+=1
    print("Length of tests: ", test_len)
    for i in range(test_len):
        print("Length of train: ", masterDataSet.trainX.shape[0], " and test: ", masterDataSet.testX.shape[0])
        if test_set: mod=None
        mod = getModelAndTrain(masterDataSet, mod, test_set)
        modPredict(mod, masterDataSet)
        if remainder != 0 and i == (test_len - 1): masterDataSet.rotateDatasets(remainder)
        masterDataSet.rotateDatasets()
    viz.displayKCrossVal(masterDataSet)


if __name__ == "__main__":
    if 'test_set' in sys.argv:
        print("========= TEST SET =========")
        if len(sys.argv) > 3: openAndTrain(True, load_datasets=True)
        else: openAndTrain(True)
    else:
        print("========= TEST SITE =========")
        if 'mod' in sys.argv: openAndTrain(False, mod='mod')
        else: openAndTrain(False)
