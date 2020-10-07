import csv
import matplotlib.pyplot as plt
import matplotlib
import glob
import os
import re
import numpy as np
import seaborn as sns
import math
import scipy.stats as stats
from utils import remove_outliers
import time
matplotlib.use('TkAgg')
import pandas as pd
from matplotlib import rc

sns.set_theme(style="whitegrid")
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=14)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')

def evaluate():
    folder = '/home/bch_svt/cartpole/simulation/localization_errors_with_quaternions'
    filename_prefix = 'localization_errors_10822*'
    all_file_names = os.listdir(folder)
    pattern = re.compile(filename_prefix)

    # get all files with errors
    file_names = []
    for f in all_file_names:
        if pattern.match(f):
            file_names.append(f)

    x_errors = []
    y_errors = []
    z_errors = []
    x_rot_errors = []
    y_rot_errors = []
    z_rot_errors = []
    a = []
    b = []
    c = []
    d = []
    heights = []

    data_dicts = []
    data_dict2 = []
    for f in file_names:

        with open(folder + '/' + f) as file:
            print(file.name)
            reader = csv.reader(file)
            data = list(reader)
            for i, row in enumerate(data):
                if i > 1000:
                    break
                # print(d)
                if int(row[6]) == 2:
                    x_errors.append(float(row[0]))
                    y_errors.append(float(row[1]))
                    z_errors.append(float(row[2]))
                    x_rot_errors.append(float(row[3]))
                    y_rot_errors.append(float(row[4]))
                    z_rot_errors.append(float(row[5]))
                    heights.append(float(row[8]))
                    a.append(float(row[9]))
                    b.append(float(row[10]))
                    c.append(float(row[11]))
                    d.append(float(row[12]))
                    data_dicts.append({'x_pos': float(row[0]),
                                       'y_pos': float(row[1]),
                                       'z_pos': float(row[2]),
                                       'x_rot': float(row[3]),
                                       'y_rot': float(row[4]),
                                       'z_rot': float(row[5]),
                                       'height': float(row[8]),
                                       'a': float(row[9]),
                                       'b': float(row[10]),
                                       'c': float(row[11]),
                                       'd': float(row[12])})
                    # data_dict2.append({'axis': 'x_pos', 'value': float(row[0]), 'height': float(row[8])})
                    # data_dict2.append({'axis': 'y_pos', 'value': float(row[1]), 'height': float(row[8])})
                    # data_dict2.append({'axis': 'z_pos', 'value': float(row[2]), 'height': float(row[8])})
                    # data_dict2.append({'axis': 'x_rot', 'value': float(row[3]), 'height': float(row[8])})
                    # data_dict2.append({'axis': 'y_rot', 'value': float(row[4]), 'height': float(row[8])})
                    # data_dict2.append({'axis': 'z_rot', 'value': float(row[5]), 'height': float(row[8])})
                    # data_dict2.append({'axis': 'a', 'value': float(row[9]), 'height': float(row[8])})
                    # data_dict2.append({'axis': 'b', 'value': float(row[10]), 'height': float(row[8])})
                    # data_dict2.append({'axis': 'c', 'value': float(row[11]), 'height': float(row[8])})
                    # data_dict2.append({'axis': 'd', 'value': float(row[12]), 'height': float(row[8])})




    # concert in degrees
    # x_rot_errors = list(map(math.degrees, x_rot_errors + list(map(lambda x: -x, x_rot_errors))))
    # y_rot_errors = list(map(math.degrees, y_rot_errors + list(map(lambda x: -x, y_rot_errors))))
    # z_rot_errors = list(map(math.degrees, z_rot_errors + list(map(lambda x: -x, z_rot_errors))))

    heights = np.array(heights)
    x_errors = np.array(x_errors)
    y_errors = np.array(y_errors)
    z_errors = np.array(z_errors)

    lvls = []
    for h in heights:
        lvls.append(calculate_lvl_from_height(h))

    print(f"Unique lvls: {np.unique(lvls)}")

    # remove outliers
    x_errors, idxs = remove_outliers(x_errors)
    heights = heights[idxs]
    y_errors = y_errors[idxs]
    z_errors = z_errors[idxs]
    y_errors, idxs = remove_outliers(y_errors)
    heights = heights[idxs]
    x_errors = x_errors[idxs]
    z_errors = z_errors[idxs]
    z_errors, idxs = remove_outliers(z_errors)
    heights = heights[idxs]
    y_errors = y_errors[idxs]
    x_errors = x_errors[idxs]
    # a, idxs = remove_outliers(a)
    # heights = heights[idxs]
    # b, idxs = remove_outliers(b)
    # heights = heights[idxs]
    # c, idxs = remove_outliers(c)
    # heights = heights[idxs]
    # d, idxs = remove_outliers(d)
    # heights = heights[idxs]
    # x_rot_errors, idxs = np.degrees(remove_outliers(x_rot_errors))
    # heights = heights[idxs]
    # y_rot_errors, idxs = np.degrees(remove_outliers(y_rot_errors))
    # heights = heights[idxs]
    # z_rot_errors, idxs = np.degrees(remove_outliers(z_rot_errors))
    # heights = heights[idxs]

    for e, h in zip(x_errors, heights):
        data_dict2.append({'axis': 'x', 'value': e, 'lvl': calculate_lvl_from_height(h)})
    for e, h in zip(y_errors, heights):
        data_dict2.append({'axis': 'y', 'value': e, 'lvl': calculate_lvl_from_height(h)})
    for e, h in zip(z_errors, heights):
        data_dict2.append({'axis': 'z', 'value': e, 'lvl': calculate_lvl_from_height(h)})
    # for e, h in zip(x_rot_errors, heights):
    #     data_dict2.append({'axis': 'x', 'value': e, 'height': h})
    # for e, h in zip(y_rot_errors, heights):
    #     data_dict2.append({'axis': 'y', 'value': e, 'height': h})
    # for e, h in zip(z_rot_errors, heights):
    #     data_dict2.append({'axis': 'z', 'value': e, 'height': h})
    # for e, h in zip(a, heights):
    #     data_dict2.append({'axis': 'q_a', 'value': e, 'height': h})
    # for e, h in zip(b, heights):
    #     data_dict2.append({'axis': 'q_b', 'value': e, 'height': h})
    # for e, h in zip(c, heights):
    #     data_dict2.append({'axis': 'q_c', 'value': e, 'height': h})
    # for e, h in zip(d, heights):
    #     data_dict2.append({'axis': 'q_d', 'value': e, 'height': h})

    # fig_dims = (7, 2)
    # fig, ax = plt.subplots(figsize=fig_dims)
    # df = pd.DataFrame(data_dict2)
    #
    # ax = sns.boxplot(ax=ax, x='value', y='axis', data=df, orient='h', showfliers=False)
    # ax.set(ylabel='Achse', xlabel='Fehler in $^\circ$')
    # plt.tight_layout()
    # plt.savefig('/home/bch_svt/masterarbeit/figures/localization_errors/rot_errors_boxplot.pdf', format='pdf')
    # plt.show()
    #
    # exit()


    fig_dims = (7, 2)
    fig, ax = plt.subplots(figsize=fig_dims)
    df = pd.DataFrame(data_dict2)

    print('Hi!')
    ax = sns.stripplot(x='lvl', y='value', hue='axis', ax=ax, data=df, dodge=True, size=3)
    # ax = sns.scatterplot(x='height', y='value', hue='axis', ax=ax, data=df)
    ax.set(ylabel='Fehler in mm', xlabel='Ebene')
    plt.legend(title='Achse')
    plt.tight_layout()
    plt.show()

    exit()



    print(f"X errors std: {np.std(x_errors)}")
    print(f"Y errors std: {np.std(y_errors)}")
    print(f"Z errors std: {np.std(z_errors)}")

    # plot all distibutions
    # plt.subplot(4, 2, 1)
    # sns.distplot(x_errors, bins=200)
    #
    # plt.subplot(4, 2, 2)
    # sns.distplot(y_errors, bins=200)
    #
    # plt.subplot(4, 2, 3)
    # sns.distplot(z_errors, bins=200)
    #
    # plt.subplot(4, 2, 4)
    # sns.distplot(a, bins=200)




    # plot histograms
    errors = x_errors


    sns.distplot(errors, bins=100)
    # sns.scatterplot(x_errors, b)

    # compute parameters for all features
    # dist = stats.norm
    # dist = stats.cauchy
    # dist = stats.hypsecant
    dist = stats.norminvgauss
    print(f"Start fit")
    # params_x = dist.fit(x_errors)
    # params_y = dist.fit(y_errors)
    # params_z = dist.fit(z_errors)
    # params_a = dist.fit(a)
    # params_b = dist.fit(b)
    # params_c = dist.fit(c)
    # params_d = dist.fit(d)
    # params_x_rot = dist.fit(x_rot_errors)
    # params_y_rot = dist.fit(y_rot_errors)
    # params_z_rot = dist.fit(z_rot_errors)
    print(f"End fit")
    # print(f"Params x: {params_x}")
    # print(f"Params y: {params_y}")
    # print(f"Params z: {params_z}")
    # print(f"Params a: {params_a}")
    # print(f"Params b: {params_b}")
    # print(f"Params c: {params_c}")
    # print(f"Params d: {params_d}")

    # params = params_d

    mu = np.mean(errors)
    sigma = np.std(errors)
    print(f"Mean: {mu:.2f}, std: {sigma:.2f}")

    # compute log likelihood of the fitted dist
    # loglh = dist.logpdf(errors, params[0], params[1], params[2], params[3]).sum()
    # print(f"Log likelihood: {loglh}")

    # compute fitted dist
    x = np.linspace(-3*sigma, 3*sigma, 100)
    # plt.plot(x, dist.pdf(x, params[0], params[1], params[2], params[3]))
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # plot fitted dist
    # plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.savefig('test_figure.pdf', format='pdf')
    plt.show()


def calculate_lvl_from_height(height):
    min_height = 0.3
    spacing = 0.75
    lvl = int((height - min_height) / spacing)

    return lvl





if __name__ == "__main__":
    evaluate()