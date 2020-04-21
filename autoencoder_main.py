import numpy as np
import random
import sys
import time
import os
import pandas as pd


from lib import rawdata
from lib import dataset
from lib import metrics
from lib import viz
from lib import preprocess
from lib import util
from lib import model
from multiprocessing import Pool


classify = True
bin_class = False
rand = False



def openDatasets(test_set, mod):
    data = []
    if mod is None:
        data = rawdata.RawData.load(locNames='all', special_layers='all')
    masterDataSet = dataset.Squares(data, test_set, mod)
    if not test_set: #its the test site
        new_data = rawdata.RawData.load(locNames='untrain', special_layers='all', new_data='not_none')
        testSiteDataset = dataset.Squares(new_data, test_set, mod=None)
        masterDataSet.testX = testSiteDataset.squares
        masterDataSet.testy = testSiteDataset.square_labels
    return masterDataSet

def getModelAndTrain(masterDataSet, mod):
    if mod is None:
        mod = model.unet(masterDataSet)
        mod.fit(masterDataSet.trainX, masterDataSet.trainy, batch_size=32, epochs=1, verbose=1)
        time_string = time.strftime("%Y%m%d-%H%M%S")
        fname = 'models/' + time_string + '_UNET-test_site.h5'
        print("Saving: ", fname)
        mod.save_weights(fname)
    else:
        mod = model.unet(masterDataSet, pretrained_weights='models/20200420-230825_UNET-test_site.h5')
    return mod

def modPredict(mod, masterDataSet):
    y_preds = mod.predict(masterDataSet.testX)
    util.evaluateUNET(y_preds, masterDataSet)

def openAndTrain(test_set=True, mod=None, train_dataset=None, val_dataset=None, test_dataset=None):
    start_time = time.time()
    from lib import model
    masterDataSet = openDatasets(test_set, mod)
    mod = getModelAndTrain(masterDataSet, mod)
    modPredict(mod, masterDataSet)


if __name__ == "__main__":
    if 'test_set' in sys.argv:
        print("========= TEST SET =========")
        openAndTrain(True)
    else:
        print("========= TEST SITE =========")
        if 'mod' in sys.argv:
            openAndTrain(False, mod='mod')
        else:
            openAndTrain(False)

# if len(sys.argv) == 5:
#     print("Making picture from predictions")
#     # python3 main.py all Test_Site_0320-170405_ [predictions] picture
#     example(sys.argv[3])
# elif len(sys.argv) == 4:
#     print('Training a new model with old datasets...')
#     # python3 main.py train0325-220335_ validate0325-220529_ test0325-220528_ | tee output_aqua.out&
#     openAndTrain(sys.argv[1], sys.argv[2], sys.argv[3])
# elif len(sys.argv) == 3:
#     print('Making test picture with model...')
#     # command: python3 main.py all Test_Site_0320-170405_
#     example()
# elif len(sys.argv) == 2:
#     # command: python3 main.py model_name
#     print('Showing model...')
#     showModel(sys.argv[1])
# elif len(sys.argv) == 1:
#     # command: python3 main.py
#     print('Training a new model...')
#     openAndTrain()
# else:
#     # command: python3 main.py make a new dataset now
#     print('making new dataset')
#     test = makeNewAreaDataset()
#     dataset.Dataset.save(test)
