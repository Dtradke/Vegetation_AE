import numpy as np
import cv2
# from scipy.misc import imsave
# from scipy.ndimage import imread
# from libtiff import TIFF
from time import localtime, strftime
import time
import csv


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

def checkNeighborhood(pred):
    # [1:-1,1:-1] vert, hor
    cur_pred = np.expand_dims(pred, axis=0)
    hor_pad = np.full((pred.shape[1], ), -1)
    vert_pad = np.full((pred.shape[0], 1), -1)
    full_pred = []

    for i in range(3):
        for j in range(3):
            if i == 0:
                cur_pred = cur_pred[:,1:] #cut top
                cur_pred = np.vstack((cur_pred, hor_pad)) #add bottom
            if j == 0:
                cur_pred = cur_pred[1:,:] #cut left
                cur_pred = np.concatenate((cur_pred, vert_pad), axis=1) # add right
            if i == 2:
                cur_pred = cur_pred[:,:-1] # cut bottom
                cur_pred = np.vstack((hor_pad, cur_pred)) # add top
            if j == 2:
                cur_pred = cur_pred[:-1,:] #cut right
                cur_pred = np.concatenate((vert_pad, cur_pred), axis=1) # add left
            full_pred.append(np.subtract(pred, cur_pred))

    print(len(full_pred))
    for pred in full_pred:
        pred[pred==0] = -1000
    # preds = [p[p==0] = -1000 for p in full_pred]
    full_pred = np.sum(full_pred, axis=0)
    full_pred[full_pred > -500] == 0

    correct = np.count_nonzero(full_pred)
    incorrect = preds.size - correct
    print("square Correct: ", correct)
    print("square Incorrect: ", incorrect)
    return correct, incorrect




def evaluateUNET(y_preds, masterDataSet):
    incorrect = 0
    correct = 0

    for i, val in enumerate(masterDataSet.testy):
        pred = y_preds[i]
        pred[pred < 0.33] = 0
        pred[(pred >= 0.33) & (pred < 0.66)] = 0.5
        pred[pred >= 0.66] = 1
        sq_correct, sq_incorrect = checkNeighborhood(pred)
        correct+=sq_correct
        incorrect+=sq_incorrect

    print("Correct: ", correct / (correct+incorrect))
    print("Incorrect: ", incorrect / (correct+incorrect))
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
