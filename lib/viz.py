from time import localtime, strftime
import csv
from mpl_toolkits.mplot3d import axes3d
import scipy
# import seaborn as sns


import numpy as np
import cv2
import random
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    print('Successfully imported pyplot')
except:
    print('Failed to import pyplot ')

from matplotlib import colors
from lib import dataset
from lib import util

classify = False
bin_class = False

small_obj_heights = False

def view3d(layer):
    print(layer)
    print()
    print()
    if layer.shape[0] >= layer.shape[1]: cut = layer.shape[1]
    else: cut = layer.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0,cut,1)
    y = np.arange(0,cut,1)
    X,Y = np.meshgrid(x,y)
    Z = layer[:cut,:cut]
    print(Z)
    dem3d=ax.plot_surface(X,Y, Z,cmap='afmhot', linewidth=0)
    ax.set_title('RESULT')
    ax.set_zlabel('Vegetation Height (m)')
    plt.show()


def viewResult(layer, val, pred, diff, r_squared, num):
    titles = ['layer', 'val', 'pred', 'diff']
    arr = [layer, val, pred, diff]
    count = 0

    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2

    ax = []
    for i in range(columns*rows):
        img = np.squeeze(arr[count])
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        # ax[-1].set_title(titles[count] + ": " + str(round((val.size - np.count_nonzero(diff)) / val.size, 4)))  # set title
        ax[-1].set_title(titles[count] + ": " + str(round(r_squared, 4)))
        plt.imshow(img) #, alpha=0.25
        count+=1
    # plt.show()
    if save:
        fname = "output/figures/" + str(num) + ".png"
        plt.savefig(fname, dpi=fig.dpi)
    plt.close()

def makeCDFclasses(stats):
    plt.style.use('ggplot')
    plt.figure(figsize=(8,4))
    # labels = ["0-2", "2-6", "6-20", "6-50", "20-50", "50-80", "80+"]
    # labels = ["0-2", "2-6", "6-20", "20-50", "50-80", "80-250"]
    labels = ["0.0 - 0.6", "0.6 - 1.8", "1.8 - 6.0", "6.0 - 15.3", "15.3 - 24.4", "24.4 +"]
    for i, stat in enumerate(stats):
        error = np.sort(np.absolute(np.subtract(stat[0], stat[1])))
        y = np.linspace(0,error.max(),error.shape[0])
        y /= y.max()
        # new_error = error[error <= np.quantile(error, 0.95)]
        # error = new_error[new_error >= np.quantile(error, 0.05)]
        # norm_error = error / error.sum()
        # cumsum_error = np.cumsum(norm_error)
        # cumsum_error = scipy.stats.norm.cdf(error)
        # x = np.linspace(0, error[-1], error.shape[0])
        plt.plot(error, y, label=labels[i])
    plt.xlim(left = 0)
    plt.ylim(bottom=0)
    plt.ylabel("Percent of Predictions (%)", fontsize=20)
    plt.xlabel("Absolute Error (m)", fontsize=20)
    plt.legend(loc='best')
    plt.title("Y-NET CDF - Ranges", fontsize=20)

    fname = "output/figures/CDF_classes.png"
    plt.savefig(fname,bbox_inches='tight', dpi=300)
    plt.close()

def makeCDFreg(y_pred, ground):
    plt.style.use('ggplot')
    plt.figure(figsize=(8,4))
    error = np.sort(np.absolute(np.subtract(y_pred, ground)))
    y = np.linspace(0,error.max(),error.shape[0])
    y /= y.max()
    # new_error = error[error <= np.quantile(error, 0.95)]
    # error = new_error[new_error >= np.quantile(error, 0.05)]
    # np.save('ynet_error.npy', error)
    # np.save('ynet_y_pred.npy', y_pred)
    # np.save('ynet_ground.npy', ground)

    # id_str = "noFoot"
    # np.save('CDF_x_'+ id_str +'.npy', error)
    # np.save('CDF_y_'+ id_str +'.npy', y)
    print("Median: ", np.median(error))
    print("AVG: ", np.mean(error))
    # norm_error = error / error.sum()
    # cumsum_error = np.cumsum(norm_error)
    # cumsum_error = scipy.stats.norm.cdf(error)
    # x = np.linspace(0, error[-1], error.shape[0])
    plt.plot(error, y)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.ylabel("Percent of Predictions (%)", fontsize=20)
    plt.xlabel("Absolute Error (m)", fontsize=20)
    # plt.legend(loc='best')
    plt.title("Y-NET CDF", fontsize=20)

    fname = "output/figures/CDF_reg.png"
    plt.savefig(fname,bbox_inches='tight', dpi=300)
    plt.close()

def viewResultColorbar(layer, val, pred, diff, r_squared=0, num=0):
    titles = ['layer', 'diff', 'val', 'pred']
    arr = [layer, diff, val, pred]
    np.random.seed(19680801)
    Nr = 2
    Nc = 2
    cmap = "viridis"

    fig, axs = plt.subplots(Nr, Nc)
    fig.suptitle("R^2: " + str(r_squared))

    images = []
    count = 0
    for i in range(Nr):
        for j in range(Nc):
            # Generate data with a range that varies from one plot to the next.
            data = arr[count]  #((1 + i + j) / 10) * np.random.rand(10, 20) * 1e-6
            if len(data.shape) > 2:
                data = np.squeeze(data)
            images.append(axs[i, j].imshow(data, cmap=cmap))
            # axs[i, j].label_outer()
            count+=1

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(np.amin(val), np.amin(pred)) #min(image.get_array().min() for image in images)
    vmax = max(np.amax(val), np.amax(pred)) #max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for i, im in enumerate(images):
        if titles[i] != 'layer':
            im.set_norm(norm)

    cb = fig.colorbar(images[-1], ax=axs, orientation='horizontal', fraction=.1)
    # cb.ax.tick_params(labelsize=20)
    # cb.tick_params(labelsize=20)

    if save:
        fname = "output/figures/norm" + str(num) + ".png"
        plt.savefig(fname, dpi=fig.dpi)
    plt.close()

def viewFullResultColorbar(val, pred, diff, num=0):
    titles = ['diff', 'val', 'pred']
    arr = [diff, val, pred]
    np.random.seed(19680801)
    Nr = 1
    Nc = 3
    cmap = "viridis"

    fig, axs = plt.subplots(Nr, Nc)
    # fig.suptitle("viz " + str(r_squared))

    images = []
    count = 0
    for i in range(Nr):
        for j in range(Nc):
            # Generate data with a range that varies from one plot to the next.
            data = arr[count]  #((1 + i + j) / 10) * np.random.rand(10, 20) * 1e-6
            if len(data.shape) > 2:
                data = np.squeeze(data)
            images.append(axs[i, j].imshow(data, cmap=cmap))
            # axs[i, j].label_outer()
            count+=1

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(np.amin(val), np.amin(pred)) #min(image.get_array().min() for image in images)
    vmax = max(np.amax(val), np.amax(pred)) #max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for i, im in enumerate(images):
        if titles[i] != 'layer':
            im.set_norm(norm)

    cb = fig.colorbar(images[-1], ax=axs, orientation='horizontal', fraction=.1)
    # cb.ax.tick_params(labelsize=20)
    # cb.tick_params(labelsize=20)

    if save:
        fname = "output/figures/full_test_loc" + str(num) + ".png"
        plt.savefig(fname, dpi=fig.dpi)
    plt.close()


def densityPlot(preds, ground):
    grid = np.zeros((250,250))

    print("pred nonzero: ", np.count_nonzero(preds), " out of: ", preds.size)
    exit()

    for g in range(250):
        print("g: ", g, " amt: ", np.count_nonzero(np.around(ground,0) == g))
        for p in range(250):
            print("p: ", p, " amt: ", np.count_nonzero(np.around(preds,0) == p))
            grid[g,p]+=np.count_nonzero((np.around(ground,0) == g) & (np.around(preds,0) == p))

    np.save('ynet_grid.npy', grid)


def scatterplotRegression(preds, ground):
    import matplotlib.lines as mlines
    error = np.absolute(np.subtract(preds, ground))
    # keep_idx = (error >= np.quantile(error,0.05)) & (error <= np.quantile(error,0.95))
    # error_cut = error[keep_idx]
    # preds_cut = preds[keep_idx]
    # ground_cut = ground[keep_idx]
    #
    # error_comp = error[~keep_idx]
    # preds_comp = preds[~keep_idx]
    # ground_comp = ground[~keep_idx]
    #
    # plt.scatter(preds_cut, ground_cut, s=0.2, c='b', alpha=0.01)
    # plt.scatter(preds_comp, ground_comp, s=0.2, c='m', alpha=0.01)
    #
    # x = np.arange(250)
    # y = np.arange(250)
    # plt.plot(x,y,c='r')
    # # line = mlines.Line2D([0, 250], [0, 250], color='red')
    #
    # m, b = np.polyfit(preds, ground, 1)
    # plt.plot(preds, m*preds + b, c='g')

    from scipy.stats import gaussian_kde
    xy = np.vstack([preds,ground])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = preds[idx], ground[idx], z[idx]
    # fig, ax = plt.subplots()
    plt.scatter(x, y, c=z, s=50, edgecolor='')


    # plt.add_line(line)
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('Lidar', fontsize=20)
    if save:
        if cut:
            fname = "output/figures/scatterplot_cut.png"
        else:
            fname = "output/figures/scatterplot.png"
        plt.savefig(fname)
    plt.close()


def displayKCrossVal(dataset):
    print("TOTALS:")
    total = 0
    for i in dataset.total.keys():
        total += dataset.total[i]
        print("key ", i, ": ", dataset.total[i])

    print("CORRECT:")
    correct = 0
    for i in dataset.correct.keys():
        correct+=dataset.correct[i]
        print("key ", i, ": ", dataset.correct[i], " PERC: ", round(((dataset.correct[i]/dataset.total[i])*100), 4))

    print("SUMMARY:")
    print("Correct: ",  correct / total)
    print("Incorrect: ", (total - correct) / total)

    print("Finished")
    exit()

def norm_predictions(preds):
    max_pred = 0
    min_pred = 1000
    normed = [None] * len(preds)
    idx = 0

    for pred in preds:
        if pred < min_pred:
            min_pred = pred
        if pred > max_pred:
            max_pred = pred


    for pred in preds:
        p = (pred-min_pred)/(max_pred - min_pred)
        normed[idx] = p
        idx = idx + 1


    return normed


def renderPredictions(dataset, predictions, preResu):
    heightpreds = {}
    count = 0
    for pt, pred in predictions.items():
        locName, location = pt
        loc = locName
        if loc not in heightpreds:
            heightpreds[loc] = []

        if classify:
            if count == 0:
                pair = (location, 0)
                count = count + 1
            else:
                pair = (location, (pred + 1))
        else:
            pair = (location, float(pred))

        heightpreds[loc].append(pair)


    results = {}
    for locName, locsAndPreds in heightpreds.items():
        locs, preds = zip(*locsAndPreds)
        xs,ys = zip(*locs)
        norm_pred = norm_predictions(preds)
        preds = [pred+1 for pred in norm_pred]
        loc = dataset.data.locs[locName]
        canvas = np.zeros(loc.layerSize, dtype=np.float32)
        canvas[(xs,ys)] = np.array(preds, dtype=np.float32)

        results[locName] = canvas
    return results

def get_error(neighborhood, pred):
    neighborhood = np.subtract(neighborhood, pred)
    return np.absolute(neighborhood)
    # return abs(val_height - pred)

def get_allowable_error(val_height):

    if small_obj_heights:
        four = 4/150
        return_small = 1/150
        return_tall = (.25 * val_height)/150
    else:
        four = 4
        return_small = 1
        return_tall = (.25 * val_height)

    if val_height < four:
        return return_small
    else:
        return return_tall

def evaluate_heights(predictions, dataset):
    current_locName = ""

    underTwo_acc = 0
    underTwo_total = 0

    twoFive_acc = 0
    twoFive_total = 0

    fiveTwenty_acc = 0
    fiveTwenty_total = 0

    twentyFifty_acc = 0
    twentyFifty_total = 0

    fifty75_acc = 0
    fifty75_total = 0
    fiftyPlus_acc = 0
    fiftyPlus_total = 0

    seven5Hund_acc = 0
    seven5Hund_total = 0

    hundPlus_acc = 0
    hundPlus_total = 0

    underFive_total = 0
    underFive_acc = 0

    fiveTen_acc = 0
    tenTwenty_acc = 0
    twentyThirty_acc = 0
    thirtyFourty_acc = 0
    fourtyFifty_acc = 0


    fiveTen_total = 0
    tenTwenty_total = 0
    twentyThirty_total = 0
    thirtyFourty_total = 0
    fourtyFifty_total = 0

    acc = 0
    not_acc = 0
    location_names = []
    total_error = 0

    total_pred = len(predictions)


    if small_obj_heights:
        two = 2/150
        five = 5/150
        ten = 10/150
        twenty = 20/150
        thirty = 30/150
        fourty = 40/150
        fifty = 50/150
        seven_five = 75/150
        hund = 100/150
    else:
        two = 2
        five = 5
        ten = 10
        twenty = 20
        thirty = 30
        fourty = 40
        fifty = 50
        seven_five = 75
        hund = 100

    count_it = 0
    predictions_result = {}

    for pt, pred in predictions.items():
        locName, location = pt

        if locName != current_locName:
            loc = dataset.data.locs[locName]
            current_locName = locName
            location_names.append(locName)


        if classify:
            if bin_class:

                x_loc, y_loc = location

                neighborhood = []
                for i in range(x_loc-1, x_loc+2):
                    window = []
                    for j in range(y_loc-1, y_loc+2):
                        window.append(np.argmax(loc.obj_height_classification[(i,j)]))
                    neighborhood.append(window)
                neighborhood = np.array(neighborhood)


                count_it = count_it + 1
                val_height_arr = loc.obj_height_classification[location]
                val_height = np.argmax(val_height_arr+1)# + 1
                # pred_height = np.argmax(pred)# + 1
                pred_height = pred
                # print(val_height, " ", pred_height)

                # print('neighborhood: ', neighborhood)
                # print('val_height_arr: ', val_height_arr)
                # print('val_height: ', val_height)
                # print('predicted height: ', pred_height)

                if val_height == 0:
                    underFive_total += 1
                else:                               #this is 5-20
                    fiveTwenty_total += 1


                # if val_height == pred_height:
                if pred_height in neighborhood.flatten():
                    predictions_result[pt] = 0
                    acc = acc + 1
                    if val_height == 0:
                        # print("under 10")
                        underFive_acc += 1
                    else:
                        # print("over 10")
                        fiveTwenty_acc += 1
                else:
                    predictions_result[pt] = 100
                    # print("not accurate")
                    not_acc = not_acc + 1
            else:
                x_loc, y_loc = location

                neighborhood = []
                for i in range(x_loc-1, x_loc+2):
                    window = []
                    for j in range(y_loc-1, y_loc+2):
                        window.append(np.argmax(loc.obj_height_classification[(i,j)]))
                    neighborhood.append(window)
                neighborhood = np.array(neighborhood)

                val_height_arr = loc.obj_height_classification[location]
                # print('val_height_arr: ', val_height_arr)
                val_height = np.argmax(val_height_arr)
                # print('val_height: ', val_height)

                # pred_height = np.argmax(pred)
                pred_height = pred
                # print('predicted height: ', pred_height, ' val_height: ', val_height)

                if val_height == 0:
                    underFive_total += 1
                elif val_height == 1:                               #this is 5-20
                    fiveTwenty_total += 1
                elif val_height == 2:                               #this is 20-50
                    twentyFifty_total += 1
                else:
                    fiftyPlus_total += 1

                # if val_height == pred_height:
                # print('flat neighborhood: ', neighborhood.flatten())
                if pred_height in neighborhood.flatten():
                    predictions_result[pt] = 1
                    acc = acc + 1
                    if val_height == 0:
                        underFive_acc += 1
                    elif val_height == 1:
                        fiveTwenty_acc += 1
                    elif val_height == 2:
                        twentyFifty_acc += 1
                    else:
                        fiftyPlus_acc += 1
                else:
                    predictions_result[pt] = 100
                    not_acc = not_acc + 1
        else:
            x_loc, y_loc = location

            neighborhood = []
            for i in range(x_loc-1, x_loc+2):
                window = []
                for j in range(y_loc-1, y_loc+2):
                    window.append(loc.layer_obj_heights[(i,j)])
                neighborhood.append(window)

            if count_it == 0:
                predictions_result[pt] = 0
                count_it+=1
                continue


            neighborhood = np.array(neighborhood)
            val_height = loc.layer_obj_heights[location]
            error = get_error(neighborhood, pred)
            total_error = total_error + np.amin(error) #adds the minimum error of neighborhood
            height_min_error = neighborhood[np.unravel_index(error.argmin(), error.shape)]
            allowable_error = get_allowable_error(height_min_error)

            error = error[np.unravel_index(error.argmin(), error.shape)]

            if small_obj_heights:
                two = 2/150
                five = 5/150
                ten = 10/150
                twenty = 20/150
                thirty = 30/150
                fourty = 40/150
                fifty = 50/150
                seven_five = 75/150
                hund = 100/150
            else:
                two = 2
                five = 5
                ten = 10
                twenty = 20
                thirty = 30
                fourty = 40
                fifty = 50
                seven_five = 75
                hund = 100

            # print(val_height)

            if val_height < two:
                underTwo_total = underTwo_total + 1
            elif val_height < five:
                twoFive_total = twoFive_total + 1
            elif val_height < ten:
                fiveTen_total = fiveTen_total + 1
            elif val_height < twenty:
                tenTwenty_total = tenTwenty_total + 1
            elif val_height < thirty:
                twentyThirty_total = twentyThirty_total + 1
            elif val_height < fourty:
                thirtyFourty_total = thirtyFourty_total + 1
            elif val_height < fifty:
                fourtyFifty_total = fourtyFifty_total + 1
            elif val_height < seven_five:
                fifty75_total = fifty75_total + 1
            elif val_height < hund:
                seven5Hund_total = seven5Hund_total + 1
            else:
                hundPlus_total = hundPlus_total + 1

            if error < allowable_error:
                predictions_result[pt] = 1
                print(pt)
                acc = acc + 1
                if val_height < two:
                    underTwo_acc = underTwo_acc + 1
                elif val_height < five:
                    twoFive_acc = twoFive_acc + 1
                elif val_height < ten:
                    fiveTen_acc = fiveTen_acc + 1
                elif val_height < twenty:
                    tenTwenty_acc = tenTwenty_acc + 1
                elif val_height < thirty:
                    twentyThirty_acc = twentyThirty_acc + 1
                elif val_height < fourty:
                    thirtyFourty_acc = thirtyFourty_acc + 1
                elif val_height < fifty:
                    fourtyFifty_acc = fourtyFifty_acc + 1
                elif val_height < seven_five:
                    fifty75_acc = fifty75_acc + 1
                elif val_height < hund:
                    seven5Hund_acc = seven5Hund_acc + 1
                else:
                    hundPlus_acc = hundPlus_acc + 1
            else:
                predictions_result[pt] = 100
                not_acc = not_acc + 1

    if classify:
        if bin_class:
            print('Percent accurate:           ', acc/total_pred)
            print('Average error in feet:      ', total_error/total_pred)
            print('Percent < 10 accurate:       ', underFive_acc/underFive_total)
            print('Percent x >= 10 accurate:    ', fiveTwenty_acc/fiveTwenty_total)
            print('Total accurate:              ', acc)
            print('Total not accurate:          ', not_acc)

            results = {'perc_accurate': acc/total_pred,
                        'avg_error_feet': total_error/total_pred,
                        'percent_acc_under10': underFive_acc/underFive_total,
                        'percent_acc_over10': fiveTwenty_acc/fiveTwenty_total,
                        'total_accurate': acc,
                        'total_not_acc': not_acc,
                        'locations': location_names}
        else:
            print('Percent accurate:           ', acc/total_pred)
            print('Average error in feet:      ', total_error/total_pred)
            print('Percent < 5 accurate:       ', underFive_acc/underFive_total)
            print('Percent 5 < x < 20 accurate:', fiveTwenty_acc/fiveTwenty_total)
            print('Percent 20 < x < 50 accurate:', twentyFifty_acc/twentyFifty_total)
            print('Percent 50 <= accurate:       ', fiftyPlus_acc/fiftyPlus_total)
            print('Total accurate:              ', acc)
            print('Total not accurate:          ', not_acc)

            results = {'perc_accurate': acc/total_pred,
                        'avg_error_feet': total_error/total_pred,
                        'percent_acc_under5': underFive_acc/underFive_total,
                        'percent_acc_5-20': fiveTwenty_acc/fiveTwenty_total,
                        'percent_acc_20-50': twentyFifty_acc/twentyFifty_total,
                        'percent_acc_50plus': fiftyPlus_acc/fiftyPlus_total,
                        'total_accurate': acc,
                        'total_not_acc': not_acc,
                        'locations': location_names}
    else:
        try:
            print('Percent accurate:           ', acc/total_pred)
            print('Average error in feet:      ', total_error/total_pred)
            print('Percent < 2 accurate:       ', underTwo_acc/underTwo_total)
            print('Percent 2 < x < 5 accurate: ', twoFive_acc/twoFive_total)
            print('Percent 5 < x < 10 accurate:', fiveTen_acc/fiveTen_total)
            print('Percent 10 < x < 20 accurate:', tenTwenty_acc/tenTwenty_total)
            print('Percent 20 < x < 30 accurate:', twentyThirty_acc/twentyThirty_total)
            print('Percent 30 < x < 40 accurate:', thirtyFourty_acc/thirtyFourty_total)
            print('Percent 40 < x < 50 accurate:', fourtyFifty_acc/fourtyFifty_total)
            print('Percent 50 < x < 75 accurate:', fifty75_acc/fifty75_total)
            print('Percent 75 < x <100 accurate:', seven5Hund_acc/seven5Hund_total)
            print('Percent 100 < accurate:      ', hundPlus_acc/hundPlus_total)
            print('Total accurate:              ', acc)
            print('Total not accurate:          ', not_acc)
        except:
            print('Percent accurate:           ', acc/total_pred)
            print('Average error in feet:      ', total_error/total_pred)
            print("There was a division by zero")

        results = {'perc_accurate': acc/total_pred,
                    'avg_error_feet': total_error/total_pred,
                    'percent_acc_under2': underTwo_acc/acc,
                    'percent_acc_2-5': twoFive_acc/acc,
                    'percent_acc_5-10': fiveTen_acc/acc,
                    'percent_acc_10-20': tenTwenty_acc/acc,
                    'percent_acc_20-30': twentyThirty_acc/acc,
                    'percent_acc_30-40': thirtyFourty_acc/acc,
                    'percent_acc_40-50': fourtyFifty_acc/acc,
                    'percent_acc_50-75': fifty75_acc/acc,
                    'percent_acc_75-100': seven5Hund_acc/acc,
                    'percent_acc_100plus': hundPlus_acc/acc,
                    'total_accurate': acc,
                    'total_not_acc': not_acc,
                    'locations': location_names}

    return results, predictions_result



def createCanvases(dataset):
    result = {}
    for locName in dataset.getUsedLocNames():
        print(locName)
        loc = dataset.data.locs[locName]
        l = loc.loadVeg(loc.name)
        h,w = l.shape
        normedDEM = util.normalize(loc.loadVeg2())

        canvas = cv2.cvtColor(normedDEM, cv2.COLOR_GRAY2RGB)
        im2, startContour, hierarchy = cv2.findContours(l.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im2, endContour, heirarchy = cv2.findContours(l.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        result[locName] = canvas
    return result

def overlay(predictionRenders, canvases):
    result = {}
    for locName in sorted(canvases):
        canvas = canvases[locName].copy()
        render = predictionRenders[locName]
        yellowToRed = np.dstack((np.ones_like(render), 1-(render-1), np.zeros_like(render)))
        canvas[render>1] = yellowToRed[render>1]
        result[locName] = canvas
    return result

def visualizePredictions(dataset, predictions, preResu):
    predRenders = renderPredictions(dataset, predictions, preResu)
    canvases = createCanvases(dataset)
    overlayed = overlay(predRenders, canvases)
    return overlayed

def showPredictions(predictionsRenders):
    locs = {}
    for locName, render in predictionsRenders.items():
        if locName not in locs:
            locs[locName] = []
        locs[locName].append(render)

    for locName, frameList in locs.items():
        frameList.sort()
        fig = plt.figure(locName, figsize=(11, 7))
        ims = []
        pos = (30,30)
        color = (0,0,1.0)
        size = 1
        thickness = 2
        for render in frameList:
            withTitle = render.copy()
            cv2.putText(render, locName, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness=thickness)
            im = plt.imshow(render)
            ims.append([im])
        anim = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                repeat_delay=0)

        def createMyOnKey(anim):
            def onKey(event):
                if event.key == 'right':
                    anim._step()
                elif event.key == 'left':
                    saved = anim._draw_next_frame
                    def dummy(a,b):
                        pass
                    anim._draw_next_frame = dummy
                    for i in range(len(anim._framedata)-2):
                        anim._step()
                    anim._draw_next_frame = saved
                    anim._step()
                elif event.key =='down':
                    anim.event_source.stop()
                elif event.key =='up':
                    anim.event_source.start()
            return onKey

        fig.canvas.mpl_connect('key_press_event', createMyOnKey(anim))
        plt.show()


def show(*imgs, imm=True):
    try:
        for i, img in enumerate(imgs):
            plt.figure(i, figsize=(8, 6))
            plt.imshow(img)
        if imm:
            plt.show()
    except:
        print("Not able to show because plt not imported")

def save(img, name):
    fname = 'output/imgs/{}.png'.format(name)
    cv2.imwrite(fname, img)

def saveModel(model, mod_string):
    from keras.utils import plot_model
    timeString = strftime("%d%b%H:%M", localtime())
    fname = 'output/modelViz/{}_{}.png'.format(timeString, mod_string)
    plot_model(model, to_file=fname, show_shapes=True)
