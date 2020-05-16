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


def saveExperiment(mod, masterDataSet, test_set, SPLIT):
    time_string = time.strftime("%Y%m%d-%H%M%S")
    if SPLIT: time_string = "YNET_" + time_string
    else: time_string = "UNET_" + time_string
    if test_set:
        if bin_class: fname = 'models/' + time_string + '_test_set_BIN.h5'
        elif classify: fname = 'models/' + time_string + '_test_set_CLASS.h5'
        else: fname = 'models/' + time_string + '_test_set_NORM.h5'
        saveDatasets(masterDataSet, fname)
    else:
        if bin_class: fname = 'models/' + time_string + '_test_site_BIN.h5'
        elif classify: fname = 'models/' + time_string + '_test_site_CLASS.h5'
        else: fname = 'models/' + time_string + '_test_site_NORM.h5'
    print("Saving: ", fname)
    # mod.save_weights(fname)
    mod.save(fname)

def saveDatasets(masterDataSet, fname):
    ''' saves formatted datasets to directory '''
    fname = fname[7:-3]
    print("Saving datasets: ", fname)
    np.save('output/datasets/' + fname + masterDataSet.data.names + 'trainX.npy', masterDataSet.trainX)
    np.save('output/datasets/' + fname + masterDataSet.data.names + 'trainy.npy', masterDataSet.trainy)
    np.save('output/datasets/' + fname + masterDataSet.data.names + 'valX.npy', masterDataSet.valX)
    np.save('output/datasets/' + fname + masterDataSet.data.names + 'valy.npy', masterDataSet.valy)
    np.save('output/datasets/' + fname + masterDataSet.data.names + 'testX.npy', masterDataSet.testX)
    np.save('output/datasets/' + fname + masterDataSet.data.names + 'testy.npy', masterDataSet.testy)

def loadDatasets(load_datasets):
    ''' Loads formatted datasets from directory '''
    print("Loading Datasets: ", load_datasets)
    files = ['trainX.npy', 'trainy.npy', 'valX.npy', 'valy.npy', 'testX.npy', 'testy.npy']
    datasets = []
    for suffix in files:
        datasets.append(np.load('output/datasets/' + load_datasets + suffix))
    return datasets

def loadSquareDatasets(load_datasets):
    ''' Loads preprocessed squares '''
    print("Loading Dataset Squares: ", load_datasets)
    files = ['squares.npy', 'labels.npy', 'labels_orig.npy']
    datasets = []

    for suffix in files:
        datasets.append(np.load('output/raw_squares/' + load_datasets + suffix))
    return datasets

def KCross(masterDataSet):
    test_len = (masterDataSet.trainX.shape[0] // masterDataSet.testX.shape[0])+1
    remainder = 0
    if (masterDataSet.trainX.shape[0] % masterDataSet.testX.shape[0]) != 0:
        remainder = (masterDataSet.trainX.shape[0] % masterDataSet.testX.shape[0])
        test_len+=1
    print("remainder: ", remainder)
    print("Length of tests: ", test_len)
    return test_len



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

    grass_close, shrub_close = 0,0 #getClosePreds(real_height, val, answers_counter, masterDataSet)

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

    grass_close, shrub_close = 0,0#getClosePreds(real_height, val, answers, masterDataSet)

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
        try: real_height = masterDataSet.orig_testy[i]
        except: real_height = np.array([])
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

        close_grass, close_shrub = 0,0#getClosePreds(real_height, val, diff, masterDataSet)
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
    # print("Close predictions would add: -2: ", (total_grass_close/(correct+incorrect)), " -1: ", (total_shrub_close/(correct+incorrect)), " total: ", ((total_grass_close+total_shrub_close)/(correct+incorrect)))
    print("Neighborhoods fast:")
    print("fast - Correct: ", ncorrect / (ncorrect+nincorrect))
    print("fast - Incorrect: ", nincorrect / (ncorrect+nincorrect))

    for i in correct_val_fast.keys():
        print("correct ", i, ": ", correct_val_fast[i], " PERC: ", (correct_val_fast[i]/total_val[i]))

    # print("Close predictions would add: -2: ", (total_fast_grass_close/(ncorrect+nincorrect)), " -1: ", (total_fast_shrub_close/(ncorrect+nincorrect)), " total: ", ((total_fast_grass_close+total_fast_shrub_close)/(correct+incorrect)))

    print("Neighborhoods slow:")
    print("slow - Correct: ", ck_correct_total / (ck_correct_total+ck_incorrect_total))
    print("slow - Incorrect: ", ck_incorrect_total / (ck_correct_total+ck_incorrect_total))

    for i in correct_val_slow.keys():
        print("correct ", i, ": ", correct_val_slow[i], " PERC: ", (correct_val_slow[i]/total_val[i]))
    # print("Close predictions would add: grass/shrub: ", (total_slow_grass_close/(ck_correct_total+ck_incorrect_total)), " shrub/tree: ", (total_slow_shrub_close/(ck_correct_total+ck_incorrect_total)), " total: ", ((total_slow_grass_close+total_slow_shrub_close)/(ck_correct_total+ck_incorrect_total)))

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
