import numpy as np
import cv2
# from scipy.misc import imsave
# from scipy.ndimage import imread
# from libtiff import TIFF
from time import localtime, strftime
import time
import csv
import sys

classify = False
bin_class = False

try:
    import matplotlib.pyplot as plt
except:
    pass

# def openImg(fname):
#     if "/special_layers/" in fname:
#         img = cv2.imread(fname, 0)
#     else:
#         img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
#
#     try:
#         img = img.astype(np.float32)
#     except AttributeError:
#         raise ValueError("Could not open the file {} as an image".format(fname))
#     channels = cv2.split(img)
#     for c in channels:
#         c[invalidPixelIndices(c)] = np.nan
#     return cv2.merge(channels)

def saveExperiment(mod, masterDataSet, test_set):
    time_string = time.strftime("%Y%m%d-%H%M%S")
    if test_set:
        if bin_class: fname = 'models/' + time_string + '_UNET-test_set_BIN.h5'
        elif classify: fname = 'models/' + time_string + '_UNET-test_set_CLASS.h5'
        else: fname = 'models/' + time_string + '_UNET-test_set_NORM.h5'
        saveDatasets(masterDataSet, fname)
    else:
        if bin_class: fname = 'models/' + time_string + '_UNET-test_site_BIN.h5'
        elif classify: fname = 'models/' + time_string + '_UNET-test_site_CLASS.h5'
        else: fname = 'models/' + time_string + '_UNET-test_site_NORM.h5'
    print("Saving: ", fname)
    mod.save_weights(fname)

def saveDatasets(masterDataSet, fname):
    print("Saving datasets")
    if bin_class: mode = '_BIN'
    elif classify: mode = '_CLASS'
    else: mode = '_NORM'
    fname = fname[7:-3]
    np.save('output/datasets/' + fname + 'trainX' + mode + '.npy', masterDataSet.trainX)
    np.save('output/datasets/' + fname + 'trainy' + mode + '.npy', masterDataSet.trainy)
    np.save('output/datasets/' + fname + 'testX' + mode + '.npy', masterDataSet.valX)
    np.save('output/datasets/' + fname + 'testy' + mode + '.npy', masterDataSet.valy)
    np.save('output/datasets/' + fname + 'testX' + mode + '.npy', masterDataSet.testX)
    np.save('output/datasets/' + fname + 'testy' + mode + '.npy', masterDataSet.testy)

def loadDatasets():
    print("Loading Datasets")
    files = ['inX.npy', 'iny.npy', 'alX.npy', 'aly.npy', 'stX.npy', 'sty.npy']
    count = 0
    datasets = []
    for fname in sys.argv:
        if fname[-7:] == files[count]:
            datasets.append(np.load('output/datasets/' + fname))
            count+=0
    return datasets


# Global dicts for results
correct_val_slow = {"footprint":0, "grass":0, "shrub":0, "tree":0}
correct_val_fast = {"footprint":0, "grass":0, "shrub":0, "tree":0}

def checkNeighborhood(pred, val):
    global correct_val_fast

    # [1:-1,1:-1] cuts hor, vert
    val_y = np.squeeze(val)
    cur_pred = np.squeeze(pred)
    pred = cur_pred
    full_pred = []

    for i in range(3):
        for j in range(3):
            cur_pred = pred
            if i == 0:
                cur_pred = cur_pred[1:,:] #cut top
                hor_pad = np.full((cur_pred.shape[1], ), -1)
                cur_pred = np.vstack((cur_pred, hor_pad)) #add bottom
            if j == 0:
                cur_pred = cur_pred[:,1:] #cut left
                vert_pad = np.full((cur_pred.shape[0], 1), -1)
                cur_pred = np.concatenate((cur_pred, vert_pad), axis=1) # add right
            if i == 2:
                cur_pred = cur_pred[:-1,:] # cut bottom
                hor_pad = np.full((cur_pred.shape[1], ), -1)
                cur_pred = np.vstack((hor_pad, cur_pred)) # add top
            if j == 2:
                cur_pred = cur_pred[:,:-1] #cut right
                vert_pad = np.full((cur_pred.shape[0], 1), -1)
                cur_pred = np.concatenate((vert_pad, cur_pred), axis=1) # add left
            full_pred.append(np.subtract(val_y, cur_pred))

    import sys
    np.set_printoptions(threshold=sys.maxsize)
    # print(full_pred[0])
    # exit()

    answers_counter = np.ones_like(full_pred[0])
    for pred in full_pred:
        for i, row in enumerate(pred):
            for j, entry in enumerate(row):
                if entry == 0:
                    answers_counter[i][j] = 0

    correct_val_fast["footprint"]+=np.count_nonzero((answers_counter == 0) & (val_y == 0))
    correct_val_fast["grass"]+=np.count_nonzero((answers_counter == 0) & (val_y == 1))
    correct_val_fast["shrub"]+=np.count_nonzero((answers_counter == 0) & (val_y == 2))
    correct_val_fast["tree"]+=np.count_nonzero((answers_counter == 0) & (val_y == 3))

    correct = np.count_nonzero((answers_counter == 0) & (val_y != 0))
    incorrect = np.count_nonzero((answers_counter != 0) & (val_y != 0))

    # incorrect = np.count_nonzero(answers_counter)
    # correct = val.size - incorrect
    # print("square Correct: ", correct)
    # print("square Incorrect: ", incorrect)
    return correct, incorrect


def slowCheckNeighborhood(pred, val):
    global correct_val_slow

    val = np.squeeze(val)
    answers = np.ones_like(val)
    diff = 1

    for i, row in enumerate(val):
        for j, entry in enumerate(row):
            for iter_i in range(3):
                for iter_j in range(3):
                    try:
                        diff = entry - pred[(i-1)+iter_i][(j-1)+iter_j]
                        if diff == 0:
                            answers[i][j] = 0
                    except:
                        pass

    correct_val_slow["footprint"]+=np.count_nonzero((answers == 0) & (val == 0))
    correct_val_slow["grass"]+=np.count_nonzero((answers == 0) & (val == 1))
    correct_val_slow["shrub"]+=np.count_nonzero((answers == 0) & (val == 2))
    correct_val_slow["tree"]+=np.count_nonzero((answers == 0) & (val == 3))

    correct = np.count_nonzero((answers == 0) & (val != 0))
    incorrect = np.count_nonzero((answers != 0) & (val != 0))
    return correct, incorrect

def formatPreds(pred, val):

    #NOTE: added for softmax (commented above)
    if classify or bin_class:
        max_pred = np.argmax(pred, axis=2)
        max_val = np.argmax(val, axis=2)
        return max_pred, max_val
    else:
        pred[pred < 0.25] = 0
        pred[(pred >= 0.25) & (pred < .5)] = 1
        pred[(pred >= 0.5) & (pred < 0.66)] = 2
        pred[pred >= 0.66] = 3
        val = np.squeeze(val)
        print("before: ", val[0])
        val[val == 0] = 0
        val[val == 0.33] = 1
        val[val == 0.66] = 2
        val[val == 1] = 3
        print("after: ", val[0])
        return pred, val

def evaluateUNET(y_preds, masterDataSet):
    # global correct_val_slow
    # global correct_val_fast
    incorrect = 0
    correct = 0

    nincorrect = 0
    ncorrect = 0
    ck_correct_total = 0
    ck_incorrect_total = 0

    total_val = {"footprint":0, "grass":0, "shrub":0, "tree":0}

    for i, val in enumerate(masterDataSet.testy):
        pred = y_preds[i]
        pred, val = formatPreds(pred, val)

        total_val["footprint"]+=np.count_nonzero(val == 0)
        total_val["grass"]+=np.count_nonzero(val == 1)
        total_val["shrub"]+=np.count_nonzero(val == 2)
        total_val["tree"]+=np.count_nonzero(val == 3)

        sq_correct, sq_incorrect = checkNeighborhood(pred, val)
        ck_correct, ck_incorrect = slowCheckNeighborhood(pred, val)
        ncorrect+=sq_correct
        nincorrect+=sq_incorrect
        ck_correct_total+=ck_correct
        ck_incorrect_total+=ck_incorrect
        diff = np.subtract(pred, val)
        correct+=np.count_nonzero((diff == 0) & (val != 0))
        incorrect+= np.count_nonzero((diff != 0) & (val != 0))

        # viz.viewResult(masterDataSet.testX[i][:, :, 2], val, pred, diff)

    print("foot: ", total_val["footprint"])
    print("grass: ", total_val["grass"])
    print("shrub: ", total_val["shrub"])
    print("tree: ", total_val["tree"])

    print("Correct: ", correct / (correct+incorrect))
    print("Incorrect: ", incorrect / (correct+incorrect))
    print("Neighborhoods:")
    print("n - Correct: ", ncorrect / (ncorrect+nincorrect))
    print("n - Incorrect: ", nincorrect / (ncorrect+nincorrect))
    print("foot: ", correct_val_fast["footprint"])
    print("grass: ", correct_val_fast["grass"])
    print("shrub: ", correct_val_fast["shrub"])
    print("tree: ", correct_val_fast["tree"])
    try:
        print("foot: ", correct_val_fast["footprint"] / total_val["footprint"], " grass: ", correct_val_fast["grass"] / total_val["grass"], " shrub: ", correct_val_fast["shrub"] / total_val["shrub"], " tree: ", correct_val_fast["tree"] / total_val["tree"])
    except:
        print("foot: ", correct_val_fast["footprint"] / total_val["footprint"], " below 10: ", correct_val_fast["grass"] / total_val["grass"], " above 10: ", correct_val_fast["shrub"] / total_val["shrub"])

    print("Neighborhoods check:")
    print("n - Correct: ", ck_correct_total / (ck_correct_total+ck_incorrect_total))
    print("n - Incorrect: ", ck_incorrect_total / (ck_correct_total+ck_incorrect_total))
    print("foot: ", correct_val_slow["footprint"])
    print("grass: ", correct_val_slow["grass"])
    print("shrub: ", correct_val_slow["shrub"])
    print("tree: ", correct_val_slow["tree"])
    try:
        print("foot: ", correct_val_slow["footprint"] / total_val["footprint"], " grass: ", correct_val_slow["grass"] / total_val["grass"], " shrub: ", correct_val_slow["shrub"] / total_val["shrub"], " tree: ", correct_val_slow["tree"] / total_val["tree"])
    except:
        print("foot: ", correct_val_slow["footprint"] / total_val["footprint"], " below 10: ", correct_val_slow["grass"] / total_val["grass"], " above 10: ", correct_val_slow["shrub"] / total_val["shrub"])


    exit()


def saveImg(fname, img):
    to_save = np.array(img.astype('float32'))
    print('datatype of saveimg is ', to_save.dtype)
    max_float = np.finfo(np.float32).max
    print("to save shape is", to_save.shape)
    print('max is ', max_float)
    to_save[np.where(np.isnan(to_save))] = max_float
    print('after conversions', to_save)
    if 'landsat' in fname:
        print('landsat to_save shape is ', to_save.shape)
    tiff = TIFF.open(fname, mode='w')
    tiff.write_image(to_save)
    tiff.close()

def validPixelIndices(layer):
    validPixelMask = 1-invalidPixelMask(layer)
    return np.where(validPixelMask)

def invalidPixelIndices(layer):
    return np.where(invalidPixelMask(layer))

def invalidPixelMask(layer):
    # If there are any massively valued pixels, just return those
    HUGE = 1e10
    huge = np.absolute(layer) > HUGE
    if np.any(huge):
        return huge

    # floodfill in from every corner, all the NODATA pixels are the same value so they'll get found
    h,w = layer.shape[:2]
    noDataMask = np.zeros((h+2,w+2), dtype = np.uint8)
    fill = 1
    seeds = [(0,0), (0,h-1), (w-1,0), (w-1,h-1)]
    for seed in seeds:
        cv2.floodFill(layer.copy(), noDataMask, seed, fill)

    # extract ouf the center of the mask, which corresponds to orig image
    noDataMask = noDataMask[1:h+1, 1:w+1]
    return noDataMask

def normalize(arr, axis=None):
    '''Rescale an array so that it varies from 0-1.

    if axis=0, then each column is normalized independently
    if axis=1, then each row is normalized independently'''

    arr = arr.astype(np.float32)
    res = arr - np.nanmin(arr, axis=axis)
    # where dividing by zero, just use zero
    res = np.divide(res, np.nanmax(res, axis=axis), out=np.zeros_like(res), where=res!=0)
    return res

def partition(things, ratios=None):
    if ratios is None:
        ratios = [.5]
    beginIndex = 0
    ratios.append(1)
    partitions = []
    for r in ratios:
        endIndex = int(round(r * len(things)))
        section = things[beginIndex:endIndex]
        partitions.append(section)
        beginIndex = endIndex
    return partitions

def saveEvaluation(results, dataset, fname=None):
    directory = 'output/results/'

    timeString = time.strftime("%m%d-%H%M%S")
    res_str = 'results_at_'
    fname = directory + '{}{}.csv'.format(res_str, timeString)
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for key, val in results.items():
            row = [str(key), str(val)]
            writer.writerow(row)

def savePredictions(predictions, mod_string, fname=None):
    directory = 'output/predictions/'
    if fname is None:
        timeString = time.strftime("%m%d-%H%M%S")
        fname = directory + '{}_{}.csv'.format(timeString, mod_string)
    if not fname.startswith(directory):
        fname = directory + fname
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for pt, pred in predictions.items():
            locName, location = pt
            y,x = location
            row = [str(locName), str(y), str(x), str(pred)]
            writer.writerow(row)

def openPredictions(fname):
    result = {}
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            locName, y, x, pred = row
            # ensure the date is 4 chars long
            x = int(x)
            y = int(y)
            pred = float(pred)
            p = dataset.Point(locName, (y,x))
            result[p] = pred
    return result


if __name__ == '__main__':
    import glob
    import matplotlib.pyplot as plt
    import random
    folder = 'data/**/perims/'
    types = ('*.tif', '*.png') # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(folder+files))
    for f in files_grabbed:
        print(f)
        img = openImg(f)
        plt.figure(f)
        if len(img.shape) > 2 and img.shape[2] > 3:
            plt.imshow(img[:,:,:3])
        else:
            plt.imshow(img)
        plt.show()
