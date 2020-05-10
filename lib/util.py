import numpy as np
import cv2
# from scipy.misc import imsave
# from scipy.ndimage import imread
# from libtiff import TIFF
from time import localtime, strftime
import time
import csv
import sys
from lib import viz

classify = True
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

def saveExperiment(mod, masterDataSet, test_set, SPLIT):
    time_string = time.strftime("%Y%m%d-%H%M%S")
    if SPLIT: time_string = "SPLIT_" + time_string
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



def checkNeighborhood(pred, val, real_height, masterDataSet, keys):
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


    for i in range(keys):
        correct_val_fast[i]+=np.count_nonzero((answers_counter == 0) & (val_y == i))

    grass_close, shrub_close = getClosePreds(real_height, val, answers_counter, masterDataSet)

    correct = np.count_nonzero((answers_counter == 0) & (val_y != 0))
    incorrect = np.count_nonzero((answers_counter != 0) & (val_y != 0))

    return correct, incorrect, grass_close, shrub_close


def slowCheckNeighborhood(pred, val, real_height, masterDataSet, keys):
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

    for i in range(keys):
        correct_val_slow[i]+=np.count_nonzero((answers == 0) & (val == i))

    grass_close, shrub_close = getClosePreds(real_height, val, answers, masterDataSet)

    correct = np.count_nonzero((answers == 0) & (val != 0))
    incorrect = np.count_nonzero((answers != 0) & (val != 0))
    return correct, incorrect, grass_close, shrub_close

def formatPreds(pred, val):


    #NOTE: added for softmax (commented above)
    if classify or bin_class:
        max_pred = np.argmax(pred, axis=2)
        max_val = np.argmax(val, axis=2)
        return max_pred, max_val
    else:
        pred = np.squeeze(pred)
        pred[pred >= 0.80] = 4
        pred[(pred >= 0.6) & (pred < 0.8)] = 3
        pred[(pred >= 0.4) & (pred < 0.6)] = 2
        pred[(pred >= 0.2) & (pred < .4)] = 1
        pred[pred < 0.2] = 0
        val = np.squeeze(val)
        val[val == 1] = 3
        val[val == 0.75] = 3
        val[val == 0.5] = 2
        val[val == 0.25] = 1
        val[val == 0] = 0
        return pred, val

def getClosePreds(real_height, val, diff, masterDataSet):
    if classify:
        wrong_heights = real_height[(diff != 0) & (val != 0)]
        grass_diff = np.absolute(np.subtract(wrong_heights, masterDataSet.split[-2]))
        close_grass = grass_diff[grass_diff < 5].size
        shrub_diff = np.absolute(np.subtract(wrong_heights, masterDataSet.split[-1]))
        close_shrub = shrub_diff[shrub_diff < 5].size
    elif bin_class:
        wrong_heights = real_height[(diff != 0) & (val != 0)]
        grass_diff = np.absolute(np.subtract(wrong_heights, masterDataSet.split[1]))
        close_grass = grass_diff[grass_diff < 5].size
        close_shrub = 0
    return close_grass, close_shrub


# TODO: Look at the squares which the model performs worst on
correct_val_fast = {}
correct_val_slow = {}
def evaluateYNET(y_preds, masterDataSet):
    global correct_val_fast
    global correct_val_slow
    if not classify and not bin_class:
        evaluateRegression(y_preds, masterDataSet)
    # global correct_val_slow
    # global correct_val_fast
    incorrect = 0
    correct = 0

    nincorrect = 0
    ncorrect = 0
    ck_correct_total = 0
    ck_incorrect_total = 0

    # total_val = {"footprint":0, "grass":0, "shrub":0, "tree":0, "tall_tree": 0, "tallest": 0}
    # total_val = {"footprint":0, "grass":0, "shrub":0, "tree":0}
    total_val = {}
    keys = y_preds.shape[3]
    for i in range(keys):
        total_val[i] = 0
        correct_val_fast[i] = 0
        correct_val_slow[i] = 0

    if len(masterDataSet.correct.keys()) == 0: masterDataSet.setKeys(keys)
    worst_arr_count = 0
    total_grass_close, total_shrub_close = 0, 0
    total_fast_grass_close, total_fast_shrub_close = 0, 0
    total_slow_grass_close, total_slow_shrub_close = 0, 0

    for i, val in enumerate(masterDataSet.testy):
        pred = y_preds[i]
        real_height = masterDataSet.orig_testy[i]
        pred, val = formatPreds(pred, val)

        total_val[0]+=np.count_nonzero(val == 0)
        total_val[1]+=np.count_nonzero(val == 1)
        total_val[2]+=np.count_nonzero(val == 2)
        total_val[3]+=np.count_nonzero(val == 3)
        total_val[4]+=np.count_nonzero(val == 4)
        total_val[5]+=np.count_nonzero(val == 5)

        sq_correct, sq_incorrect, fast_grass_close, fast_shrub_close = checkNeighborhood(pred, val, real_height, masterDataSet, keys)
        ck_correct, ck_incorrect, slow_grass_close, slow_shrub_close = slowCheckNeighborhood(pred, val, real_height, masterDataSet, keys)
        ncorrect+=sq_correct
        nincorrect+=sq_incorrect
        ck_correct_total+=ck_correct
        ck_incorrect_total+=ck_incorrect
        diff = np.subtract(pred, val)
        correct+=np.count_nonzero((diff == 0) & (val != 0))
        incorrect+= np.count_nonzero((diff != 0) & (val != 0))

        close_grass, close_shrub = getClosePreds(real_height, val, diff, masterDataSet)
        total_grass_close+=close_grass
        total_shrub_close+=close_shrub
        total_fast_grass_close+=fast_grass_close
        total_fast_shrub_close+=fast_shrub_close
        total_slow_grass_close+=slow_grass_close
        total_slow_shrub_close+=slow_shrub_close


        if np.count_nonzero((diff != 0) & (val != 0)) > worst_arr_count:
            worst_arr_count = np.count_nonzero((diff != 0) & (val != 0))
            worst_arr = diff
            worst_arr_pred = pred
            worst_arr_val = val

        # viz.view3d(val)
        # viz.viewResult(masterDataSet.testX[i][:, :, 2], val, pred, diff)

    # print("How many wrong preds are close: grass: ", (close_grass/(correct+incorrect)), " shrub: ", (close_shrub/(correct+incorrect)))
    # np.set_printoptions(threshold=sys.maxsize)
    # print("amt wrong: ", worst_arr_count / (64*64))
    # print("worst arr diff: ", worst_arr)
    # print()
    # print("worst arr val: ", worst_arr_val)
    # viz.viewResult(masterDataSet.testX[i][:, :, 2], worst_arr_val, worst_arr_pred, worst_arr)

    for i in total_val.keys():
        print("total ", i, ": ", total_val[i])

    print("Correct: ", correct / (correct+incorrect))
    print("Incorrect: ", incorrect / (correct+incorrect))
    print("Close predictions would add: -2: ", (total_grass_close/(correct+incorrect)), " -1: ", (total_shrub_close/(correct+incorrect)), " total: ", ((total_grass_close+total_shrub_close)/(correct+incorrect)))
    print("Neighborhoods:")
    print("fast - Correct: ", ncorrect / (ncorrect+nincorrect))
    print("fast - Incorrect: ", nincorrect / (ncorrect+nincorrect))

    for i in correct_val_fast.keys():
        print("correct ", i, ": ", correct_val_fast[i], " PERC: ", (correct_val_fast[i]/total_val[i]))


    # try:
    #     print("foot: ", correct_val_fast["footprint"] / total_val["footprint"], " grass: ", correct_val_fast["grass"] / total_val["grass"], " shrub: ", correct_val_fast["shrub"] / total_val["shrub"], " tree: ", correct_val_fast["tree"] / total_val["tree"], " tall_tree: ", correct_val_fast["tall_tree"] / total_val["tall_tree"], " tallest: ", correct_val_fast["tallest"] / total_val["tallest"])
    #     # print("foot: ", correct_val_fast["footprint"] / total_val["footprint"], " grass: ", correct_val_fast["grass"] / total_val["grass"], " shrub: ", correct_val_fast["shrub"] / total_val["shrub"], " tree: ", correct_val_fast["tree"] / total_val["tree"])
    # except:
    #     print("foot: ", correct_val_fast["footprint"] / total_val["footprint"], " below 10: ", correct_val_fast["grass"] / total_val["grass"], " above 10: ", correct_val_fast["shrub"] / total_val["shrub"])
    print("Close predictions would add: -2: ", (total_fast_grass_close/(ncorrect+nincorrect)), " -1: ", (total_fast_shrub_close/(ncorrect+nincorrect)), " total: ", ((total_fast_grass_close+total_fast_shrub_close)/(correct+incorrect)))

    print("Neighborhoods check:")
    print("slow - Correct: ", ck_correct_total / (ck_correct_total+ck_incorrect_total))
    print("slow - Incorrect: ", ck_incorrect_total / (ck_correct_total+ck_incorrect_total))

    for i in correct_val_slow.keys():
        print("correct ", i, ": ", correct_val_slow[i], " PERC: ", (correct_val_slow[i]/total_val[i]))
    # try:
    #     print("foot: ", correct_val_slow[0] / total_val["footprint"], " grass: ", correct_val_slow["grass"] / total_val["grass"], " shrub: ", correct_val_slow["shrub"] / total_val["shrub"], " tree: ", correct_val_slow["tree"] / total_val["tree"], " tall_tree: ", correct_val_slow["tall_tree"] / total_val["tall_tree"], " tallest: ", correct_val_slow["tallest"] / total_val["tallest"])
    #     # print("foot: ", correct_val_slow["footprint"] / total_val["footprint"], " grass: ", correct_val_slow["grass"] / total_val["grass"], " shrub: ", correct_val_slow["shrub"] / total_val["shrub"], " tree: ", correct_val_slow["tree"] / total_val["tree"])
    # except:
    #     print("foot: ", correct_val_slow["footprint"] / total_val["footprint"], " below 10: ", correct_val_slow["grass"] / total_val["grass"], " above 10: ", correct_val_slow["shrub"] / total_val["shrub"])
    print("Close predictions would add: grass/shrub: ", (total_slow_grass_close/(ck_correct_total+ck_incorrect_total)), " shrub/tree: ", (total_slow_shrub_close/(ck_correct_total+ck_incorrect_total)), " total: ", ((total_slow_grass_close+total_slow_shrub_close)/(ck_correct_total+ck_incorrect_total)))

    for i in correct_val_slow.keys():
        masterDataSet.correct[i] += correct_val_slow[i]
        masterDataSet.total[i] += total_val[i]



def evaluateRegression(y_preds, masterDataSet):
    # error = np.mean( y_preds != masterDataSet.testy )
    # ground = np.squeeze(masterDataSet.testy)
    # y_preds = np.squeeze(y_preds)
    ground = masterDataSet.testy.flatten()
    y_preds = y_preds.flatten()

    diff = np.absolute(np.subtract(ground, y_preds))
    below10diff = diff[ground < 10]
    between10and50 = diff[(ground >= 10) & (ground < 50)]
    above50diff = diff[ground >= 50]

    error_under10 = np.mean(below10diff)
    error_middle = np.mean(between10and50)
    error_high = np.mean(above50diff)
    print("short: ", error_under10)
    print("middle: ", error_middle)
    print("high: ", error_high)


    # percentage = np.divide(diff, ground)
    error = np.mean(np.nan_to_num(diff))
    print("Error in feet: ", error)
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
