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
import math

classify = False
bin_class = False

try:
    import matplotlib.pyplot as plt
except:
    pass


def saveExperiment(mod, masterDataSet, test_set, SPLIT):
    time_string = time.strftime("%Y%m%d-%H%M%S")
    if SPLIT: time_string = "YNET_" + time_string
    else: time_string = "UNET_" + time_string
    time_string = time_string + masterDataSet.trainstring
    if test_set:
        if bin_class: fname = 'models/' + time_string + '_test_set_BIN.h5'
        elif classify: fname = 'models/' + time_string + '_test_set_CLASS.h5'
        else: fname = 'models/' + time_string + '_test_set_NORM.h5'
        # saveDatasets(masterDataSet, fname)
    else:
        if bin_class: fname = 'models/' + time_string + '_test_site_BIN.h5'
        elif classify: fname = 'models/' + time_string + '_test_site_CLASS.h5'
        else: fname = 'models/' + time_string + '_test_site_NORM.h5'
    saveDatasets(masterDataSet, fname)
    print("Saving: ", fname)
    # mod.save_weights(fname)
    mod.save(fname)

def saveDatasets(masterDataSet, fname):
    ''' saves formatted datasets to directory '''
    fname = fname[7:-3]
    print("Saving datasets: ", fname)
    np.save('output/datasets/' + fname + 'trainX.npy', masterDataSet.trainX)
    np.save('output/datasets/' + fname + 'trainy.npy', masterDataSet.trainy)
    np.save('output/datasets/' + fname + 'train_ids.npy', masterDataSet.train_ids)
    np.save('output/datasets/' + fname + 'valX.npy', masterDataSet.valX)
    np.save('output/datasets/' + fname + 'valy.npy', masterDataSet.valy)
    np.save('output/datasets/' + fname + 'val_ids.npy', masterDataSet.val_ids)
    np.save('output/datasets/' + fname + 'testX.npy', masterDataSet.testX) #teststring
    np.save('output/datasets/' + fname + 'testy.npy', masterDataSet.testy)
    np.save('output/datasets/' + fname + 'test_ids.npy', masterDataSet.test_ids)

def loadDatasets(load_datasets, save_mod):
    ''' Loads formatted datasets from directory '''
    print("Loading Datasets: ", load_datasets)
    if save_mod: files = ['trainX.npy', 'trainy.npy', 'train_ids.npy', 'valX.npy', 'valy.npy', 'val_ids.npy', 'testX.npy', 'testy.npy', 'test_ids.npy']
    else: files = ['testX.npy', 'testy.npy', 'test_ids.npy', 'testX.npy', 'testy.npy', 'test_ids.npy', 'testX.npy', 'testy.npy', 'test_ids.npy']
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


def slowCheckNeighborhood(sample, pred, val, real_height, masterDataSet, keys, ground):
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

    wrong = pred[answers == 1]
    right = val[answers == 1]
    for i, height in enumerate(right):
        ground[height][wrong[i]]+=1

    # if sample < 500:
    #     viz.viewResult(masterDataSet.testX[i][:, :, -3], val, pred, answers, num=sample)

    grass_close, shrub_close = 0,0#getClosePreds(real_height, val, answers, masterDataSet)

    correct = np.count_nonzero((answers == 0) & (val != 0))
    incorrect = np.count_nonzero((answers != 0) & (val != 0))
    return correct, incorrect, grass_close, shrub_close, ground

def formatPreds(pred, val):
    #NOTE: added for softmax (commented above)
    if classify or bin_class:
        max_pred = np.argmax(pred, axis=2)
        max_val = np.argmax(val, axis=2)
        return max_pred, max_val
    else:
        pred = np.squeeze(pred)
        val = np.squeeze(val)
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

def makeClassification(arr):
    arr[(arr >= 0) & (arr < 2)] = 1
    arr[(arr >= 2) & (arr < 6)] = 2
    arr[(arr >= 6) & (arr < 20)] = 3
    arr[(arr >= 20) & (arr < 50)] = 4
    arr[(arr >= 50) & (arr < 80)] = 5
    arr[arr >= 80] = 6
    arr[arr < 0] = 0
    return arr

def classifyRegression(y_preds, masterDataSet):
    for i in range(masterDataSet.testy.shape[0]):
        y_preds[i] = makeClassification(y_preds[i])
        masterDataSet.testy[i] = makeClassification(masterDataSet.testy[i])
    return y_preds, masterDataSet


# TODO: Look at the squares which the model performs worst on
correct_val_fast = {}
correct_val_slow = {}
def evaluateYNET(y_preds, masterDataSet):
    global correct_val_fast
    global correct_val_slow
    if not classify and not bin_class:
        evaluateRegression(y_preds, masterDataSet)
        # y_preds, masterDataSet = classifyRegression(y_preds, masterDataSet)
        keys = 7
    else:
        keys = y_preds.shape[3]
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
    # the keys of ground are the known heights, the dict values represent the wrong predictions of that class
    ground = {}
    for i in range(keys):
        total_val[i] = 0
        correct_val_fast[i] = 0
        correct_val_slow[i] = 0
        ground[i] = {}
        for j in range(keys):
            ground[i][j] = 0


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
        total_val[6]+=np.count_nonzero(val == 6)



        sq_correct, sq_incorrect, fast_grass_close, fast_shrub_close = checkNeighborhood(pred, val, real_height, masterDataSet, keys)
        ck_correct, ck_incorrect, slow_grass_close, slow_shrub_close, ground = slowCheckNeighborhood(i, pred, val, real_height, masterDataSet, keys, ground)
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

    print("WRONG PREDICTION BREAKDOWN (for confusion mtx):")
    for i, key in enumerate(ground.keys()):
        for w, inner_key in enumerate(ground[key].keys()):
            print("Ground: ", key, " - Total predicted wrong as ", inner_key,": ", ground[key][inner_key]/total_val[key])


def calculateRSquared(y_pred, ground):
    # error = np.sort(np.absolute(np.subtract(y_pred, ground)))
    # new_error = error[error <= np.quantile(error, 0.95)]
    # error = new_error[new_error >= np.quantile(error, 0.05)]

    # RSS = np.sum(np.square(np.subtract(val, pred)))
    # TSS = np.sum(np.square(np.subtract(val,np.mean(val))))
    # r_squared = 1 - (RSS/TSS)


    error = np.absolute(np.subtract(y_pred, ground))
    # keep_idx = [(error >= np.quantile(error,0.05)) & (error <= np.quantile(error,0.95))]
    # error = error[keep_idx]
    # pred = y_pred[keep_idx]
    # val = ground[keep_idx]

    RSS = np.sum(np.square(np.subtract(ground, y_pred)))
    TSS = np.sum(np.square(np.subtract(ground,np.mean(ground))))
    r_squared = 1 - (RSS/TSS)
    return y_pred, ground, r_squared


##### YNET RESULTS
ynet_results = {0:{2:1.5178},
                2:{6:4.0433},
                6:{20:6.6985, 50:7.2223},
                20:{50:7.4397},
                50:{80:10.6338},
                80:{251:24.2859},
                "r": 0.83168,
                "median": 1.9104,
                "avg": 5.2976,
                "rmse": 10.240919}


def calcError(y_preds, ground, lower=0, upper=2):
    y_preds = y_preds[(ground >= lower) & (ground < upper)]
    ground = ground[(ground >= lower) & (ground < upper)]
    rmse = np.sqrt(np.mean(np.square(np.subtract(ground, y_preds))))
    avg_abs = np.mean(np.absolute(np.subtract(ground, y_preds)))
    print("lower: ", lower, " - upper: ", upper, " - rmse: ", rmse, " - avg_error_ft: ", avg_abs, " - median: ", np.median(avg_abs), " - amt: ", ground.size)
    print("lower: ", lower, " - upper: ", upper, " - rmse: ", rmse, " - avg_error_ft: ", (avg_abs - ynet_results[lower][upper]), " - median: ", (np.median(avg_abs) - ynet_results[lower][upper]), " - amt: ", ground.size)
    return [y_preds, ground]
    # return rmse

def evaluateRegression(y_preds, masterDataSet):
    single_r_squareds = []
    pred_squares, val_squares, img_squares = [], [], []

    full_viz_pred = masterDataSet.test_arrays
    print(full_viz_pred[0].shape)
    full_viz_ground = []
    full_viz_imagery = []
    for i in full_viz_pred:
        full_viz_ground.append(np.zeros(i.shape))
        full_viz_imagery.append(np.zeros(i.shape))
    # make visuals
    for i, val in enumerate(masterDataSet.testy):
        pred = y_preds[i]
        try: real_height = masterDataSet.orig_testy[i]
        except: real_height = np.array([])
        pred, val = formatPreds(pred, val)
        flat_pred = pred[val>0]
        flat_val = val[val>0]
        mse = np.mean(np.square(np.subtract(val, pred)))
        absolute_diff = np.absolute(np.subtract(val, pred))

        _,_,r = calculateRSquared(flat_pred, flat_val)
        if math.isnan(r):
            continue
        single_r_squareds.append(r)

        # loc = masterDataSet.test_ids[i]
        # row = int(64*loc[1])
        # col = int(64*loc[2])
        # full_viz_pred[int(loc[0])][row:int(row+pred.shape[0]), col:int(col+pred.shape[1])] += pred
        # full_viz_ground[int(loc[0])][row:int(row+val.shape[0]), col:int(col+val.shape[1])] += val
        # imagery = masterDataSet.testX[i][:, :, -3]
        # full_viz_imagery[int(loc[0])][row:int(row+imagery.shape[0]), col:int(col+imagery.shape[1])] += imagery
        # if i < 500:
        #     viz.viewResult(masterDataSet.testX[i][:, :, -3], val, pred, absolute_diff, single_r_squareds[-1], i)
        pred_squares.append(pred)
        val_squares.append(val)
        img_squares.append(masterDataSet.testX[i][:, :, -3])
        viz.viewResultColorbar(masterDataSet.testX[i][:, :, -3], val, pred, absolute_diff, single_r_squareds[-1], i)
    #

    for i, loc in enumerate(full_viz_pred):
        diff = np.absolute(np.subtract(full_viz_ground[i], full_viz_pred[i]))
        # viz.viewFullResultColorbar(full_viz_ground[i], full_viz_pred[i], diff, num=i)
        # print("pred ", i,": ", np.array(full_viz_pred[i])[0])
        # print("ground ", i,": ", np.array(full_viz_ground[i])[0])
    #     np.save('full_viz_ground'+str(i)+'.npy', np.array(full_viz_ground[i]))
    #     np.save('full_viz_pred'+str(i)+'.npy', np.array(full_viz_pred[i]))
    #     np.save('diff'+str(i)+'.npy', np.array(diff))
    #     np.save('imagery'+str(i)+'.npy', np.array(full_viz_imagery[i]))
    #
    # np.save('new_ynet_squares_pred.npy', np.array(pred_squares))
    # np.save('new_ynet_squares_ground.npy', np.array(val_squares))
    # np.save('new_ynet_squares_img.npy', np.array(img_squares))

    # calculate result
    ground = masterDataSet.testy.flatten()
    y_preds = y_preds.flatten()
    y_preds = y_preds[ground>=0]
    ground = ground[ground>=0]

    print("Median: ", np.median(np.absolute(np.subtract(y_preds, ground))) - ynet_results["median"])
    print("Mean: ", np.mean(np.absolute(np.subtract(y_preds, ground))) - ynet_results["avg"])
    print("Median: ", np.median(np.absolute(np.subtract(y_preds, ground))))
    print("Mean: ", np.mean(np.absolute(np.subtract(y_preds, ground))))

    # viz.scatterplotRegression(y_preds, ground)
    # viz.makeCDFreg(y_preds, ground)

    y_preds, ground, r_sqr = calculateRSquared(y_preds, ground)

    print("R^2 together: ", r_sqr)
    print("R^2 together: ", r_sqr - ynet_results["r"])
    print("R^2 separate: ", np.mean(np.array(single_r_squareds)))

    stats = []
    stats.append(calcError(y_preds, ground, lower=0, upper=2))
    stats.append(calcError(y_preds, ground, lower=2, upper=6))
    stats.append(calcError(y_preds, ground, lower=6, upper=20))
    stats.append(calcError(y_preds, ground, lower=6, upper=50))
    stats.append(calcError(y_preds, ground, lower=20, upper=50))
    stats.append(calcError(y_preds, ground, lower=50, upper=80))
    stats.append(calcError(y_preds, ground, lower=80, upper=251))
    # viz.makeCDFclasses(stats)

    rmse = np.sqrt(np.mean(np.square(np.subtract(ground, y_preds))))
    print("mean_squared_error: ", rmse)
    print("mean_squared_error: ", rmse - ynet_results["rmse"])
    print("Finished")
    exit()
