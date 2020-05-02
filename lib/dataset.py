import math
from collections import namedtuple
import random
import json
import time
from time import localtime, strftime
from multiprocessing import Pool
from keras.utils import to_categorical


import numpy as np
# import cv2
from lib import rawdata
from lib import viz
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

AOIRadius = 11

classify = True
bin_class = False

small_obj_heights = False

SQUARE_DIM = 64


class Squares(object):
    '''Makes a dataset of squares for the autoencoders'''

    def __init__(self, data, test_set=False, mod=None, datasets=None):
        print("mod: ", mod)
        if datasets is not None:
            self.trainX, self.trainy, self.valX, self.valy, self.testX, self.testy = datasets
        else:
            if mod is None:
                self.data = data
                self.squares, self.square_labels = self.makeSquares()
                self.square_labels_orig = self.square_labels
                # self.measureBal()
                print(self.square_labels_orig)
                print()
                self.makeClasses()
                print(self.square_labels_orig)
                exit()
                if test_set: self.trainX, self.trainy, self.orig_trainy, self.testX, self.testy, self.orig_testy = self.splitDataset()
                else: self.trainX, self.trainy, self.square_labels_orig, self.testX, self.testy = self.squares, self.square_labels, self.square_labels_orig, [], [], []
                self.makeValDataset()
                print(self.orig_testy)
                exit()
            else:
                self.testX, self.testy, self.square_labels_orig = [], [], []

    def measureBal(self):
        total = self.square_labels[0].shape[0] * self.square_labels[0].shape[1]
        print(total)
        for i, square in enumerate(self.squares):
            footprint = 0
            grass = 0
            shrub = 0
            tree = 0
            footprint = np.count_nonzero(np.argmax(self.square_labels[i], 2) == 0)
            grass = np.count_nonzero(np.argmax(self.square_labels[i], 2) == 1)
            shrub = np.count_nonzero(np.argmax(self.square_labels[i], 2) == 2)
            tree = np.count_nonzero(np.argmax(self.square_labels[i], 2) == 3)

            print(i, " - foot: ", round((footprint/total), 4)," - grass: ", round((grass/total), 4)," - shrub: ", round((shrub/total), 4)," - tree: ", round((tree/total), 4))

    def makeClasses(self):
        print(self.square_labels.shape)
        square_label = np.array(self.square_labels)
        square_label = square_label.flatten()
        sorted_squares = np.sort(square_label)
        sorted_squares = sorted_squares[sorted_squares != -1]
        if classify:
            for i in range(sorted_squares.shape[0]):
                try:
                    split_arr = np.split(sorted_squares, 3)
                    break
                except:
                    print("Popping for equal div of 4 from shape: ", sorted_squares.shape)
                    sorted_squares = sorted_squares[:-1]
            grass = split_arr[0][-1]
            shrub = split_arr[1][-1]
            tree = split_arr[2][-1]
            print("split arr: ")
            for i in split_arr:
                print(i[-1], " len: ", len(i))
            print("grass: 0 - ", grass, " shrub: ", grass, " - ", shrub, " tree: ", shrub)
            self.square_labels[(self.square_labels >= 0) & (self.square_labels <= grass)] = 1
            self.square_labels[self.square_labels == -1] = 0
            self.square_labels[(self.square_labels > grass) & (self.square_labels <= shrub)] = 2
            self.square_labels[self.square_labels > shrub] = 3
            self.square_labels = to_categorical(self.square_labels, 4)
        if bin_class:
            for i in range(sorted_squares.shape[0]):
                try:
                    split_arr = np.split(sorted_squares, 2)
                    break
                except:
                    print("Popping for equal div of 2 from shape: ", sorted_squares.shape)
                    sorted_squares = sorted_squares[:-1]
            grass = split_arr[0][-1]
            tree = split_arr[1][-1]
            self.square_labels[(self.square_labels >= 0) & (self.square_labels <= grass)] = 1
            self.square_labels[self.square_labels == -1] = 0
            self.square_labels[self.square_labels > grass] = 2
            self.square_labels = to_categorical(self.square_labels, 3)


    def makeValDataset(self):
        l = int(self.trainX.shape[0] * 0.8)
        self.valX = self.trainX[-l:]
        self.valy = self.trainy[-l:]
        self.trainX = self.trainX[:l]
        self.trainy = self.trainy[:l]

    def splitDataset(self):
        split = 0.7
        trainX = self.squares[:int(self.squares.shape[0] * split)]
        trainy = self.square_labels[:int(self.squares.shape[0] * split)]
        orig_trainy = self.square_labels_orig[:int(self.squares.shape[0] * split)]
        testX = self.squares[int(self.squares.shape[0] * split):]
        testy = self.square_labels[int(self.squares.shape[0] * split):]
        orig_testy = self.square_labels_orig[int(self.squares.shape[0] * split):]
        return trainX, trainy, orig_trainy, testX, testy, orig_testy


# TODO: ADD AUGMENTATION - rotation and offset
    def makeSquares(self):
        all_cubes = []
        all_cubes_labels = []
        for i, loc in enumerate(self.data.locs.values()):
            print("Making squares for: ", loc.name)
            layers_arr = []
            cube = []
            for layer in loc.layers.values():
                split_indices = [SQUARE_DIM*d for d in range(1,(layer.shape[1]//SQUARE_DIM)+1)]
                h_split = np.hsplit(layer, np.array(split_indices))
                last = h_split[-1]
                if last.shape[1] < SQUARE_DIM:
                    h_split.pop()
                layer_squares = []
                for slice in h_split:
                    split_indices = [SQUARE_DIM*d for d in range(1, (layer.shape[0]//SQUARE_DIM)+1)]
                    v_split = np.vsplit(slice, np.array(split_indices))
                    last = v_split[-1]
                    if last.shape[0] < SQUARE_DIM:
                        v_split.pop()
                    layer_squares = layer_squares + v_split
                layers_arr.append(np.array(layer_squares))
            cubes = np.stack(layers_arr, axis=3)
            cube_labels = self.makeLabel(loc.layer_obj_heights)
            print(cubes.shape, " labels: ", cube_labels.shape)
            all_cubes.append(cubes)
            all_cubes_labels.append(cube_labels)
        all_cubes = np.concatenate(all_cubes, axis=0 )
        all_cubes_labels = np.concatenate(all_cubes_labels, axis=0 )
        return shuffle(all_cubes, all_cubes_labels)

    @staticmethod
    def makeLabel(label_layer):
        squares = []
        split_indices = [SQUARE_DIM*d for d in range(1, (label_layer.shape[1]//SQUARE_DIM)+1)]
        h_split = np.hsplit(label_layer, np.array(split_indices))
        last = h_split[-1]
        if last.shape[1] < SQUARE_DIM:
            h_split.pop()
        for slice in h_split:
            split_indices = [SQUARE_DIM*d for d in range(1, (label_layer.shape[0]//SQUARE_DIM)+1)]
            v_split = np.vsplit(slice, np.array(split_indices))
            last = v_split[-1]
            if last.shape[0] < SQUARE_DIM:
                v_split.pop()
            if not classify and not bin_class:
                v_split = [np.expand_dims(v, axis=2) for v in v_split]
            squares = squares + v_split
        return np.array(squares)


class Dataset(object):
    '''A set of Point objects'''
    VULNERABLE_RADIUS = 500

    def __init__(self, data, points='all'):
        print('creating new dataset with points type: ', type(points))
        self.data = data

        self.points = points
        if points=='all':
            points = Dataset.allPixels
        if hasattr(points, '__call__'):
            # points is a filter function
            filterFunc = points
            self.points = self.filterPoints(self.data, filterFunc)


        if type(points) == list:
            print(len(points))
            list_split = self.split_list(points, 40)
            total_dataset_dict = {}

            cores = 40
            chunksize = 1
            with Pool(processes=cores) as pool:
                dataset_arr = pool.map(self.toDict, list_split, chunksize)

            for i in dataset_arr:
                for j in i.keys():
                    total_dataset_dict[j] = []

            for i in dataset_arr:
                for j in i.keys():
                    total_dataset_dict[j] = total_dataset_dict[j] + i[j]

            self.points = total_dataset_dict #self.toDict(points)


        the_type = type(self.points)
        comp = type({})
        assert the_type == comp

    @staticmethod
    def split_list(alist, wanted_parts=1):
        length = len(alist)
        return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
                for i in range(wanted_parts) ]

    def getUsedLocNames(self):
        results = []
        locNames = self.points.keys()
        for loc in locNames:
            results.append(loc)
        return results

    def getAllLayers(self, layerName):
        result = {}
        allLocNames = list(self.points.keys())
        for locName in allLocNames:
            loc = self.data.locs[locName]
            layer = loc.layers[layerName]
            result[locName] = layer
        return result

    def __len__(self):
        total = 0
        for locName, locDict in self.points.items():
            for ptList in locDict.values():
                total += len(ptList)
        return total

    def save(self, fname=None):
        if classify:
            if bin_class:
                timeString = time.strftime("%m%d-%H%M%S" + "classBIN")
            else:
                timeString = time.strftime("%m%d-%H%M%S" + "class")
        else:
            timeString = time.strftime("%m%d-%H%M%S" + "regress")
        if fname is None:
            fname = timeString
        else:
            fname = fname + timeString
        if not fname.startswith("output/datasets/"):
            fname = "output/datasets/" + fname
        if not fname.endswith('.json'):
            fname = fname + '_'

        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(MyEncoder, self).default(obj)

        with open(fname, 'w') as fp:
            json.dump(self.points, fp, cls=MyEncoder, sort_keys=True, indent=4)

        return fname

    @staticmethod
    def toList(pointDict):
        '''Flatten the point dictionary to a list of Points'''
        result = []
        for locName in pointDict:
            points = pointDict[locName]
            result.extend(points)
        return result


    @staticmethod
    def toListTest(pointDict):
        '''Flatten the point dictionary to a list of Points'''
        result = []
        for locName in pointDict:
            points = pointDict[locName].values()
            for p in points:
                result.extend(p)

        return result

    @staticmethod
    def toDict(pointList):
        locs = {}
        locsCount = {}
        locsFillCount = {}
        for p in pointList:
            locName, _ = p
            if locName not in locsCount:
                locsCount[locName] = 1
            else:
                current = locsCount[locName]
                current = current + 1
                locsCount[locName] = current

        for name in locsCount.keys():
            locs[name] = [None] * locsCount[name]
            locsFillCount[name] = 0

        for p in range(len(pointList)):
            locName, location = pointList[p]

            if pointList[p] not in locs[locName]:
                locs[locName][locsFillCount[locName]] = pointList[p]
                locsFillCount[locName] = locsFillCount[locName] + 1

        return locs

    @staticmethod
    def filterPoints(data, filterFunction):
        '''Return all the points which satisfy some filterFunction'''
        points = {}
        locations = data.locs.values()
        for loc in locations:
            dictOfLoc = {}
            points[loc.name] = dictOfLoc
            oh = loc.loadVeg(loc.name, 'all')

            y_x = filterFunction(loc, loc)
            dictOfLoc[loc] = [Point(loc.name,l) for l in y_x]
        return points


    def sample(self, goalNumber='max', sampleEvenly=True):
        assert goalNumber == 'max' or (type(goalNumber)==int and goalNumber%2==0)
        height_res = self.makeDay2HighLowMap()


        max_pixel_amt = [0] * 10

        if classify:
            if bin_class:
                [print(loc, " : ", len(class1), len(class2)) for loc, (class1, class2) in height_res.items()]
                limits = {loc:min(len(class1), len(class2)) for loc, (class1, class2) in height_res.items()}
            else:
                [print(loc, " : ", len(class1), len(class2), len(class3), len(class4)) for loc, (class1, class2, class3, class4) in height_res.items()]
                limits = {loc:min(len(class1), len(class2), len(class3), len(class4)) for loc, (class1, class2, class3, class4) in height_res.items()}
        else:
            # [print(loc, " : ", len(underTwo), len(twoFive), len(fiveTen), len(tenTwenty), len(twentyThirty), len(thirtyFourty), len(fourtyFifty), len(fifty75), len(seven5Hund), len(hundPlus)) for loc, (underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus) in height_res.items()]

            # for loc, height_dict in height_res.items():
            #     print(loc, " : ", height_dict.size
            [print(loc, " : ", len(height_dict[0])) for loc, height_dict in height_res.items()]

            # find the limiting size for each location
            # limits = {loc:min(len(underTwo), len(twoFive), len(fiveTen), len(tenTwenty), len(twentyThirty), len(thirtyFourty), len(fourtyFifty), len(fifty75), len(seven5Hund), len(hundPlus)) for loc, (underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus) in height_res.items()}
            limits = {}
            for loc, height_dict in height_res.items():
                location_pixel_height_amt = []
                for lst in height_dict.values():
                    location_pixel_height_amt.append(len(lst))
                limits[loc] = min(location_pixel_height_amt)

        print(limits)

        if sampleEvenly:
            # we must get the same number of samples from each day
            # don't allow a large location to have a bigger impact on training
            if goalNumber == 'max':
                # get as many samples as possible while maintaining even sampling
                samplesPerLoc = min(limits.values())
                print("samplesPerLoc", samplesPerLoc)
            else:
                # aim for a specific number of samples and sample evenly
                maxSamples = (2 * min(limits.values())) * len(limits)
                if goalNumber > maxSamples:
                    raise ValueError("Not able to get {} samples while maintaining even sampling from the available {}.".format(goalNumber, maxSamples))
                nlocs = len(limits)
                samplesPerLoc = goalNumber/(2*nlocs)
                samplesPerLoc = int(math.ceil(samplesPerLoc))
        else:
            # we don't care about sampling evenly. Larger Days will get more samples
            if goalNumber == 'max':
                # get as many samples as possible, whatever it takes
                samplesPerLoc = 'max'
            else:
                # aim for a specific number of samples and don't enforce even sampling
                samplesPerLoc = goalNumber/10
                maxSamples = sum(limits.values()) * 10
                if goalNumber > maxSamples:
                    raise ValueError("Not able to get {} samples from the available {}.".format(goalNumber, maxSamples))
        # order the days from most limiting to least limiting
        locs = sorted(limits, key=limits.get)
        tallSamples = []
        shortSamples = []
        underTwoSamples = []
        twoFiveSamples = []
        fiveTenSamples = []
        tenTwentySamples = []
        twentyThirtySamples = []
        thirtyFourtySamples = []
        fourtyFiftySamples = []
        fifty75Samples = []
        seven5HundSamples = []
        hundPlusSamples = []

        return_dict = {}
        for i in range(101):
            return_dict[i] = []

        class1Samples = []
        class2Samples = []
        class3Samples = []
        class4Samples = []

        if classify:
            if bin_class:
                class1, class2 = [], []

                for i, loc in enumerate(locs):
                    class1_loc, class2_loc = height_res[loc] #tall, short

                    class1.extend(class1_loc)
                    class2.extend(class2_loc)

                random.shuffle(class1)
                random.shuffle(class2)

                if sampleEvenly:
                    print('now samplesPerLoc', samplesPerLoc)
                    class1Samples.extend(class1[:samplesPerLoc])
                    class2Samples.extend(class2[:samplesPerLoc])
                else:
                    if samplesPerLoc == 'max':
                        nsamples = min(len(class1), len(class2))
                        class1Samples.extend(class1[:nsamples])
                        class2Samples.extend(class2[:nsamples])
                    else:
                        samplesToGo = goalNumber/2 - len(class1Samples)
                        locsToGo = len(locs)-i
                        goalSamplesPerLoc = int(math.ceil(samplesToGo/locsToGo))
                        nsamples = min(goalSamplesPerLoc,len(class1), len(class2))
                        class1Samples.extend(class1[:nsamples])
                        class2Samples.extend(class2[:nsamples])

                # now shuffle, trim and split the samples
                print('length of all samples', len(class1), len(class2))
                random.shuffle(class1Samples)
                random.shuffle(class2Samples)

                if goalNumber != 'max':
                    class1Samples = class1Samples[:goalNumber//2]
                    class2Samples = class2Samples[:goalNumber//2]

                samples = []
                train = []
                val = []
                test = []
                for i in return_dict.keys():
                    # samples = samples + return_dict[i]
                    train = class1Samples[:int(len(class1Samples) * .7)] + class2Samples[:int(len(class2Samples) * .7)]
                    val = class1Samples[int(len(class1Samples) * .7):int(len(class1Samples) * .8)] + class2Samples[int(len(class2Samples) * .7):int(len(class2Samples) * .8)]
                    test = class1Samples[int(len(class1Samples) * .8):] + class2Samples[int(len(class2Samples) * .8):]
                print('train: ', len(train))
                print('val: ', len(val))
                print('test: ', len(test))
                random.shuffle(train)
                random.shuffle(val)
                random.shuffle(test)
                return train, val, test

                # samples = class1Samples + class2Samples
            else:
                class1, class2, class3, class4 = [], [], [], []

                for i, loc in enumerate(locs):
                    class1_loc, class2_loc, class3_loc, class4_loc = height_res[loc] #tall, short

                    class1.extend(class1_loc)
                    class2.extend(class2_loc)
                    class3.extend(class3_loc)
                    class4.extend(class4_loc)

                random.shuffle(class1)
                random.shuffle(class2)
                random.shuffle(class3)
                random.shuffle(class4)

                if sampleEvenly:
                    print('now samplesPerLoc', samplesPerLoc)
                    class1Samples.extend(class1[:samplesPerLoc])
                    class2Samples.extend(class2[:samplesPerLoc])
                    class3Samples.extend(class3[:samplesPerLoc])
                    class4Samples.extend(class4[:samplesPerLoc])
                else:
                    if samplesPerLoc == 'max':
                        nsamples = min(len(class1), len(class2), len(class3), len(class4))
                        class1Samples.extend(class1[:nsamples])
                        class2Samples.extend(class2[:nsamples])
                        class3Samples.extend(class3[:nsamples])
                        class4Samples.extend(class4[:nsamples])
                    else:
                        samplesToGo = goalNumber/2 - len(class1Samples)
                        locsToGo = len(locs)-i
                        goalSamplesPerLoc = int(math.ceil(samplesToGo/locsToGo))
                        nsamples = min(goalSamplesPerLoc,len(class1), len(class2), len(class3), len(class4))
                        class1Samples.extend(class1[:nsamples])
                        class2Samples.extend(class2[:nsamples])
                        class3Samples.extend(class3[:nsamples])
                        class4Samples.extend(class4[:nsamples])

                # now shuffle, trim and split the samples
                print('length of all samples', len(class1), len(class2), len(class3), len(class4))
                # random.shuffle(tallSamples)
                # random.shuffle(shortSamples)
                random.shuffle(class1Samples)
                random.shuffle(class2Samples)
                random.shuffle(class3Samples)
                random.shuffle(class4Samples)

                if goalNumber != 'max':
                    class1Samples = class1Samples[:goalNumber//4]
                    class2Samples = class2Samples[:goalNumber//4]
                    class3Samples = class3Samples[:goalNumber//4]
                    class4Samples = class4Samples[:goalNumber//4]

                samples = []
                train = []
                val = []
                test = []
                for i in return_dict.keys():
                    # samples = samples + return_dict[i]
                    train = class1Samples[:int(len(class1Samples) * .7)] + class2Samples[:int(len(class2Samples) * .7)] + class3Samples[:int(len(class3Samples) * .7)] + class4Samples[:int(len(class4Samples) * .7)]
                    val = class1Samples[int(len(class1Samples) * .7):int(len(class1Samples) * .8)] + class2Samples[int(len(class2Samples) * .7):int(len(class2Samples) * .8)] + class3Samples[int(len(class3Samples) * .7):int(len(class3Samples) * .8)] + class4Samples[int(len(class4Samples) * .7):int(len(class4Samples) * .8)]
                    test = class1Samples[int(len(class1Samples) * .8):] + class2Samples[int(len(class2Samples) * .8):] + class3Samples[int(len(class3Samples) * .8):] + class4Samples[int(len(class4Samples) * .8):]
                print('train: ', len(train))
                print('val: ', len(val))
                print('test: ', len(test))
                random.shuffle(train)
                random.shuffle(val)
                random.shuffle(test)
                return train, val, test

                # samples = class1Samples + class2Samples + class3Samples + class4Samples

        else:
            # underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus = [], [], [], [], [], [], [], [], [], []

            reg_height_dict = {}
            for i in range(101):
                reg_height_dict[i] = []

            for i, loc in enumerate(locs):
                # underTwo_loc, twoFive_loc, fiveTen_loc, tenTwenty_loc, twentyThirty_loc, thirtyFourty_loc, fourtyFifty_loc, fifty75_loc, seven5Hund_loc, hundPlus_loc = height_res[loc] #tall, short
                height_dict = height_res[loc]

                for j in height_dict.keys():
                    reg_height_dict[j].extend(height_dict[j])

                # underTwo.extend(underTwo_loc)
                # twoFive.extend(twoFive_loc)
                # fiveTen.extend(fiveTen_loc)
                # tenTwenty.extend(tenTwenty_loc)
                # twentyThirty.extend(twentyThirty_loc)
                # thirtyFourty.extend(thirtyFourty_loc)
                # fourtyFifty.extend(fourtyFifty_loc)
                # fifty75.extend(fifty75_loc)
                # seven5Hund.extend(seven5Hund_loc)
                # hundPlus.extend(hundPlus_loc)

            for i in reg_height_dict.keys():
                random.shuffle(reg_height_dict[i])

            # random.shuffle(underTwo)
            # random.shuffle(twoFive)
            # random.shuffle(fiveTen)
            # random.shuffle(tenTwenty)
            # random.shuffle(twentyThirty)
            # random.shuffle(thirtyFourty)
            # random.shuffle(fourtyFifty)
            # random.shuffle(fifty75)
            # random.shuffle(seven5Hund)
            # random.shuffle(hundPlus)

            if sampleEvenly:
                print('now samplesPerLoc', samplesPerLoc)
                # underTwoSamples.extend(underTwo[:samplesPerLoc])
                # twoFiveSamples.extend(twoFive[:samplesPerLoc])
                # fiveTenSamples.extend(fiveTen[:samplesPerLoc])
                # tenTwentySamples.extend(tenTwenty[:samplesPerLoc])
                # twentyThirtySamples.extend(twentyThirty[:samplesPerLoc])
                # thirtyFourtySamples.extend(thirtyFourty[:samplesPerLoc])
                # fourtyFiftySamples.extend(fourtyFifty[:samplesPerLoc])
                # fifty75fifty75Samples.extend(fifty75[:samplesPerLoc])
                # seven5HundSamples.extend(seven5Hund[:samplesPerLoc])
                # hundPlusSamples.extend(hundPlus[:samplesPerLoc])
                for i in reg_height_dict.keys():
                    return_dict[i].extend(reg_height_dict[i][:samplesPerLoc])
            else:
                if samplesPerLoc == 'max':
                    # nsamples = min(len(underTwo), len(twoFive), len(fiveTen), len(tenTwenty), len(twentyThirty), len(thirtyFourty), len(fourtyFifty), len(fifty75), len(seven5Hund), len(hundPlus))
                    # underTwoSamples.extend(underTwo[:nsamples])
                    # twoFiveSamples.extend(twoFive[:nsamples])
                    # fiveTenSamples.extend(fiveTen[:nsamples])
                    # tenTwentySamples.extend(tenTwenty[:nsamples])
                    # twentyThirtySamples.extend(twentyThirty[:nsamples])
                    # thirtyFourtySamples.extend(thirtyFourty[:nsamples])
                    # fourtyFiftySamples.extend(fourtyFifty[:nsamples])
                    # fifty75Samples.extend(fifty75[:nsamples])
                    # seven5HundSamples.extend(seven5Hund[:nsamples])
                    # hundPlusSamples.extend(hundPlus[:nsamples])
                    nsamples = 100000000
                    for i in reg_height_dict.keys():
                        if len(reg_height_dict[i]) < nsamples:
                            nsamples = len(reg_height_dict[i])
                    for i in reg_height_dict.keys():
                        return_dict[i].extend(reg_height_dict[i][:nsamples])
                else:
                    # samplesToGo = goalNumber/2 - len(underTwoSamples)
                    samplesToGo = goalNumber/2 - len(reg_height_dict[0])
                    locsToGo = len(locs)-i
                    goalSamplesPerLoc = int(math.ceil(samplesToGo/locsToGo))
                    # nsamples = min(goalSamplesPerLoc,len(underTwo), len(twoFive), len(fiveTen), len(tenTwenty), len(twentyThirty), len(thirtyFourty), len(fourtyFifty), len(fifty75), len(seven5Hund), len(hundPlus))
                    # underTwoSamples.extend(underTwo[:nsamples])
                    # twoFiveSamples.extend(twoFive[:nsamples])
                    # fiveTenSamples.extend(fiveTen[:nsamples])
                    # tenTwentySamples.extend(tenTwenty[:nsamples])
                    # twentyThirtySamples.extend(twentyThirty[:nsamples])
                    # thirtyFourtySamples.extend(thirtyFourty[:nsamples])
                    # fourtyFiftySamples.extend(fourtyFifty[:nsamples])
                    # fifty75Samples.extend(fifty75[:nsamples])
                    # seven5HundSamples.extend(seven5Hund[:nsamples])
                    # hundPlusSamples.extend(hundPlus[:nsamples])
                    nsamples = goalSamplesPerLoc
                    for i in reg_height_dict.keys():
                        if len(reg_height_dict[i]) < nsamples:
                            nsamples = len(reg_height_dict[i])
                    for i in reg_height_dict.keys():
                        return_dict[i].extend(reg_height_dict[i][:nsamples])

            # now shuffle, trim and split the samples
            # print('length of all samples', len(underTwoSamples), len(twoFiveSamples), len(fiveTenSamples), len(tenTwentySamples), len(twentyThirtySamples), len(thirtyFourtySamples), len(fourtyFiftySamples), len(fifty75Samples), len(seven5HundSamples), len(hundPlusSamples))
            # random.shuffle(underTwoSamples)
            # random.shuffle(twoFiveSamples)
            # random.shuffle(fiveTenSamples)
            # random.shuffle(tenTwentySamples)
            # random.shuffle(twentyThirtySamples)
            # random.shuffle(thirtyFourtySamples)
            # random.shuffle(fourtyFiftySamples)
            # random.shuffle(fifty75Samples)
            # random.shuffle(seven5HundSamples)
            # random.shuffle(hundPlusSamples)
            print("Length of regression return arrays:")
            for i in return_dict.keys():
                print('Height: ', i, ' Samples: ', len(return_dict[i]))
                random.shuffle(return_dict[i])
            if goalNumber != 'max':
                # underTwoSamples = underTwoSamples[:goalNumber//10]
                # twoFiveSamples = twoFiveSamples[:goalNumber//10]
                # fiveTenSamples = fiveTenSamples[:goalNumber//10]
                # tenTwentySamples = tenTwentySamples[:goalNumber//10]
                # twentyThirtySamples = twentyThirtySamples[:goalNumber//10]
                # thirtyFourtySamples = thirtyFourtySamples[:goalNumber//10]
                # fourtyFiftySamples = fourtyFiftySamples[:goalNumber//10]
                # fifty75Samples = fifty75Samples[:goalNumber//10]
                # seven5HundSamples = seven5HundSamples[:goalNumber//10]
                # hundPlusSamples = hundPlusSamples[:goalNumber//10]
                print("SHOULD BE THIS AMOUNT: ", goalNumber//len(return_dict.keys()))
                for i in return_dict.keys():
                    return_dict[i] = return_dict[i][:goalNumber//len(return_dict.keys())]

                    print(len(return_dict[i]))
            # samples = underTwoSamples + twoFiveSamples + fiveTenSamples + tenTwentySamples + twentyThirtySamples + thirtyFourtySamples + fourtyFiftySamples + fifty75Samples + seven5HundSamples + hundPlusSamples
            samples = []
            train = []
            val = []
            test = []
            for i in return_dict.keys():
                # samples = samples + return_dict[i]
                train = train + return_dict[i][:int(len(return_dict[i]) * .7)]
                val = val + return_dict[i][int(len(return_dict[i]) * .7):int(len(return_dict[i]) * .8)]
                test = test + return_dict[i][int(len(return_dict[i]) * .8):]
            print('train: ', len(train))
            print('val: ', len(val))
            print('test: ', len(test))
            random.shuffle(train)
            random.shuffle(val)
            random.shuffle(test)
            return train, val, test

        random.shuffle(samples)
        print(len(samples), sum(limits.values()))

        return samples

    def makeDay2HighLowMap(self):
        result = {}
        for locName, vegDict in self.points.items():
            for layer, ptList in vegDict.items():
                if classify:
                    if bin_class:
                        heights = layer.obj_height_classification
                        class1, class2 = [], []
                        for pt in ptList:
                            _,location = pt

                            if heights[location][0] == 1:
                                class1.append(pt)
                            elif heights[location][1] == 1:
                                class2.append(pt)

                        result[(locName, 'allVeg')] = (class1, class2)
                    else:
                        heights = layer.obj_height_classification
                        class1, class2, class3, class4 = [], [], [], []
                        for pt in ptList:
                            _,location = pt

                            if heights[location][0] == 1:
                                class1.append(pt)
                            elif heights[location][1] == 1:
                                class2.append(pt)
                            elif heights[location][2] == 1:
                                class3.append(pt)
                            elif heights[location][3] == 1:
                                class4.append(pt)

                        result[(locName, 'allVeg')] = (class1, class2, class3, class4)
                else:
                    heights = layer.layer_obj_heights
                    # tall, short = [], []

                    # if small_obj_heights:
                    #     two = 2/150
                    #     five = 5/150
                    #     ten = 10/150
                    #     twenty = 20/150
                    #     thirty = 30/150
                    #     fourty = 40/150
                    #     fifty = 50/150
                    #     seven_five = 75/150
                    #     hund = 100/150
                    # else:
                    #     two = 2
                    #     five = 5
                    #     ten = 10
                    #     twenty = 20
                    #     thirty = 30
                    #     fourty = 40
                    #     fifty = 50
                    #     seven_five = 75
                    #     hund = 100

                    # underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus = [], [], [], [], [], [], [], [], [], []
                    height_dict = {}
                    for i in range(101):
                        height_dict[i] = []

                    for pt in ptList:
                        _ ,location = pt

                        if int(heights[location]) > 100:
                            height_dict[100].append(pt)
                        else:
                            height_dict[int(heights[location])].append(pt)

                        # if heights[location] < two:
                        #     underTwo.append(pt)
                        # elif heights[location] < five:
                        #     twoFive.append(pt)
                        # elif heights[location] < ten:
                        #     fiveTen.append(pt)
                        # elif heights[location] < twenty:
                        #     tenTwenty.append(pt)
                        # elif heights[location] < thirty:
                        #     twentyThirty.append(pt)
                        # elif heights[location] < fourty:
                        #     thirtyFourty.append(pt)
                        # elif heights[location] < fifty:
                        #     fourtyFifty.append(pt)
                        # elif heights[location] < seven_five:
                        #     fifty75.append(pt)
                        # elif heights[location] < hund:
                        #     seven5Hund.append(pt)
                        # else:
                        #     hundPlus.append(pt)


                    result[(locName, 'allVeg')] = height_dict #(underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus) #(tall, short)
        return result

    @staticmethod
    def allPixels(loc):
        return list(np.ndindex(loc.layerSize))

    @staticmethod
    def vulnerablePixels(loc, location, radius=VULNERABLE_RADIUS):
        '''Return the indices of the pixels that are vegetation'''
        startingVeg = location.loadVeg(location.name, 'all')

        neg_aoi = 0 - AOIRadius
        #only grabs pixels within the AOIRadius to not sample off of the edge
        startingVeg = startingVeg[AOIRadius: neg_aoi, AOIRadius: neg_aoi]
        ys, xs = np.where(startingVeg == 0) #border
        ys = ys + AOIRadius
        xs = xs + AOIRadius
        return list(zip(ys, xs))

    def __repr__(self):
        # shorten the string repr of self.points
        return "Dataset({}, with {} points)".format(self.data, len(self.toList(self.points)))

# create a class that represents a spatial and temporal location that a sample lives at
Point = namedtuple('Point', ['LocName', 'location'])

def split_list2(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
            for i in range(wanted_parts) ]


def make_newPtList(pts_arr):
    newPtList = [ None ] * len(pts_arr)
    idx = 0

    for name, loc in pts_arr:
        newPtList[idx] = Point(name, tuple(loc))
        idx = idx + 1

    return newPtList


def openDataset(fname, file_loc=None):
    print('in openDataset')
    with open(fname, 'r') as fp:
        print('in openDataset with')
        if file_loc == 'untrain':
            print('UNTRAIN')
            data = rawdata.RawData.load(locNames='untrain', special_layers='all')
        else:
            print('NOT UNTRAIN')
            data = rawdata.RawData.load(locNames='all', special_layers='all')
        print('still in with')
        pts = json.load(fp)
        newLocDict =  {}

        for locName, pts_arr in pts.items():
            print('in dataset for loop')
            print("in inner dataset for loop")
            newPtList = []


            list_split = split_list2(pts_arr, 2)


            cores = 40 #was 2
            chunksize = 1
            with Pool(processes=cores) as pool:
                newPtList_arr = pool.map(make_newPtList, list_split, chunksize)


            for i in newPtList_arr:
                newPtList.extend(i)

            newLocDict[locName] = newPtList
            print("SIZE OF PTLIST: " , len(newPtList))

        return Dataset(data, newLocDict)

if __name__ == '__main__':
    d = rawdata.RawData.load()
