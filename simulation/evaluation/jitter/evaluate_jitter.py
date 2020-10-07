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
from utils.utils import remove_outliers
import time
matplotlib.use('TkAgg')
import pandas as pd
import json
from matplotlib import rc

sns.set_theme(style="whitegrid")
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=14)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')

def evaluate():
    folder = '/home/bch_svt/cartpole/simulation/evaluation/jitter'
    filename1 = 'pos_errors_1im_april.json'
    filename2 = 'pos_errors_2im_april.json'
    filename3 = 'pos_errors_3im_april2.json'
    filename4 = 'pos_errors_1im_aruco.json'

    mean_fn1 = 'pos_errors_1im_april.json'
    mean_fn2 = 'pos_errors_2im_april.json'
    mean_fn3 = 'pos_errors_3im_april.json'
    mean_fn4 = 'pos_errors_1im_aruco.json'

    data1 = None
    with open(folder + '/' + filename1) as file:
        print(file.name)
        data1 = json.load(file)

    data2 = None
    with open(folder + '/' + filename2) as file:
        print(file.name)
        data2 = json.load(file)

    data3 = None
    with open(folder + '/' + filename3) as file:
        print(file.name)
        data3 = json.load(file)

    data4 = None
    with open(folder + '/' + filename4) as file:
        print(file.name)
        data4= json.load(file)

    for d in data1:
        d['Bilder'] = '1'
        d['Detektor'] = 'Apriltags'
        d['lvl'] = calculate_lvl_from_height(d['height'])
        d['error'] = abs(d['error'])

    for d in data2:
        d['Bilder'] = '2'
        d['Detektor'] = 'Apriltags'
        d['lvl'] = calculate_lvl_from_height(d['height'])
        d['error'] = abs(d['error'])

    for d in data3:
        d['Bilder'] = '3'
        d['Detektor'] = 'Apriltags'
        d['lvl'] = calculate_lvl_from_height(d['height'])
        d['error'] = abs(d['error'])

    for d in data4:
        d['Bilder'] = '1'
        d['Detektor'] = 'ArUco'
        d['lvl'] = calculate_lvl_from_height(d['height'])
        d['error'] = abs(d['error'])

    df = pd.DataFrame(data1 + data4)

    data_mean1 = None
    with open(folder + '/' + mean_fn1) as file:
        print(file.name)
        data_mean1 = json.load(file)

    data_mean2 = None
    with open(folder + '/' + mean_fn2) as file:
        print(file.name)
        data_mean2 = json.load(file)

    data_mean3 = None
    with open(folder + '/' + mean_fn3) as file:
        print(file.name)
        data_mean3 = json.load(file)

    data_mean4 = None
    with open(folder + '/' + mean_fn4) as file:
        print(file.name)
        data_mean4 = json.load(file)

    for d in data_mean1:
        d['Detektor'] = 'Apriltags'
        d['lvl'] = calculate_lvl_from_height(d['height'])
        d['error'] = abs(d['error'])

    # print(data_mean1)
    # print(data_mean2)
    # print(data_mean3)
    # print(data_mean4)

    for d in data_mean2:
        d['Detektor'] = 'Apriltags'
        d['lvl'] = calculate_lvl_from_height(d['height'])
        d['error'] = abs(d['error'])

    for d in data_mean3:
        d['Detektor'] = 'Apriltags'
        d['lvl'] = calculate_lvl_from_height(d['height'])
        d['error'] = abs(d['error'])

    for d in data_mean4:
        d['Detektor'] = 'ArUco'
        d['lvl'] = calculate_lvl_from_height(d['height'])
        d['error'] = abs(d['error'])

    sum_x = 0
    sum_y = 0
    sum_z = 0
    count_x = 0
    count_y = 0
    count_z = 0
    for d in data_mean1:

        print(d)
        if d['axis'] == 'x':
            sum_x += d['error']
            count_x += 1
            print(f"Hi!")

        if d['axis'] == 'y':
            sum_y += d['error']
            count_y += 1

        if d['axis'] == 'z':
            sum_z += d['error']
            count_z += 1

    print(f"Mean of April 1im X: {sum_x / count_x}")
    print(f"Mean of April 1im Y: {sum_y / count_y}")
    print(f"Mean of April 1im Z: {sum_z / count_z}")

    sum_x = 0
    sum_y = 0
    sum_z = 0
    count_x = 0
    count_y = 0
    count_z = 0
    for d in data_mean2:
        if d['axis'] == 'x':
            sum_x += d['error']
            count_x += 1

        if d['axis'] == 'y':
            sum_y += d['error']
            count_y += 1

        if d['axis'] == 'z':
            sum_z += d['error']
            count_z += 1

    print(f"Mean of April 2im X: {sum_x / count_x}")
    print(f"Mean of April 2im Y: {sum_y / count_y}")
    print(f"Mean of April 2im Z: {sum_z / count_z}")

    sum_x = 0
    sum_y = 0
    sum_z = 0
    count_x = 0
    count_y = 0
    count_z = 0
    for d in data_mean3:
        if d['axis'] == 'x':
            sum_x += d['error']
            count_x += 1

        if d['axis'] == 'y':
            sum_y += d['error']
            count_y += 1

        if d['axis'] == 'z':
            sum_z += d['error']
            count_z += 1

    print(f"Mean of April 3im X: {sum_x / count_x}")
    print(f"Mean of April 3im Y: {sum_y / count_y}")
    print(f"Mean of April 3im Z: {sum_z / count_z}")

    sum_x = 0
    sum_y = 0
    sum_z = 0
    count_x = 0
    count_y = 0
    count_z = 0
    for d in data_mean4:
        if d['axis'] == 'x':
            sum_x += d['error']
            count_x += 1

        if d['axis'] == 'y':
            sum_y += d['error']
            count_y += 1

        if d['axis'] == 'z':
            sum_z += d['error']
            count_z += 1

    print(f"Mean of ArUco 1im X: {sum_x / count_x}")
    print(f"Mean of ArUco 1im Y: {sum_y / count_y}")
    print(f"Mean of ArUco 1im Z: {sum_z / count_z}")

    # exit()





    # ax = sns.stripplot(data=df, x='lvl', y='error', hue='axis', dodge=True, size=4)
    # ax.set(ylabel='Fehler in mm', xlabel='Ebene')
    # plt.legend(title='Achse')
    # plt.tight_layout()
    # #
    # plt.show()
    #
    # exit()

    # def estimator(data):
    #     data = np.sort(data)
    #     return np.min(data[-10:])
    #
    #
    # ax = sns.barplot(data=df, x='axis', y='error', hue='Bilder', estimator=np.mean)
    # ax.set(xlabel='Achse', ylabel='Fehler in mm')
    # plt.show()

    fig_dims = (7, 2)
    fig, ax = plt.subplots(figsize=fig_dims)

    ax = sns.boxplot(ax=ax, x='error', y='axis', hue='Detektor', data=df, orient='h', showfliers=False)
    # ax.set(ylabel='Achse', xlabel='Fehler in $^\circ$')
    ax.set(ylabel='Achse', xlabel='Fehler in mm')
    ax.set(xlim=(-0.01, 1.5))
    plt.tight_layout()
    plt.savefig('/home/bch_svt/masterarbeit/figures/jitter/detection_libraries_jitter_comparison.pdf', format='pdf')
    plt.show()

    exit()

    # remove outliers
    x_errors, idxs = remove_outliers(x_errors)
    # heights = heights[idxs]
    # y_errors = y_errors[idxs]
    # z_errors = z_errors[idxs]
    y_errors, idxs = remove_outliers(y_errors)
    # heights = heights[idxs]
    # x_errors = x_errors[idxs]
    # z_errors = z_errors[idxs]
    z_errors, idxs = remove_outliers(z_errors)
    # heights = heights[idxs]
    # y_errors = y_errors[idxs]
    # x_errors = x_errors[idxs]
    a, idxs = remove_outliers(a)
    # heights = heights[idxs]
    b, idxs = remove_outliers(b)
    # heights = heights[idxs]
    c, idxs = remove_outliers(c)
    # heights = heights[idxs]
    d, idxs = remove_outliers(d)
    # heights = heights[idxs]
    x_rot_errors, idxs = remove_outliers(x_rot_errors)
    x_rot_errors = np.degrees(x_rot_errors)
    # heights = heights[idxs]
    y_rot_errors, idxs = remove_outliers(y_rot_errors)
    y_rot_errors = np.degrees(y_rot_errors)
    # heights = heights[idxs]
    z_rot_errors, idxs = remove_outliers(z_rot_errors)
    z_rot_errors = np.degrees(z_rot_errors)
    # heights = heights[idxs]


    print(f"x_error mean: {np.mean(np.abs(x_errors)):.2f}")
    print(f"y_error mean: {np.mean(np.abs(y_errors)):.2f}")
    print(f"z_error mean: {np.mean(np.abs(z_errors)):.2f}")
    print(f"x_rot_error mean: {np.mean(np.abs(x_rot_errors)):.2f}")
    print(f"y_rot_error mean: {np.mean(np.abs(y_rot_errors)):.2f}")
    print(f"z_rot_error mean: {np.mean(np.abs(z_rot_errors)):.2f}")
    print(f"x_error 95% quantile: {np.quantile(np.abs(x_errors), 0.95):.2f}")
    print(f"y_error 95% quantile: {np.quantile(np.abs(y_errors), 0.95):.2f}")
    print(f"z_error 95% quantile: {np.quantile(np.abs(z_errors), 0.95):.2f}")
    print(f"x_rot_error 95% quantile: {np.quantile(np.abs(x_rot_errors), 0.95):.2f}")
    print(f"y_rot_error 95% quantile: {np.quantile(np.abs(y_rot_errors), 0.95):.2f}")
    print(f"z_rot_error 95% quantile: {np.quantile(np.abs(z_rot_errors), 0.95):.2f}")
    print(f"x_error 99% quantile: {np.quantile(np.abs(x_errors), 0.99):.2f}")
    print(f"y_error 99% quantile: {np.quantile(np.abs(y_errors), 0.99):.2f}")
    print(f"z_error 99% quantile: {np.quantile(np.abs(z_errors), 0.99):.2f}")
    print(f"x_rot_error 99% quantile: {np.quantile(np.abs(x_rot_errors), 0.99):.2f}")
    print(f"y_rot_error 99% quantile: {np.quantile(np.abs(y_rot_errors), 0.99):.2f}")
    print(f"z_rot_error 99% quantile: {np.quantile(np.abs(z_rot_errors), 0.99):.2f}")
    print(f"x_error mean: {np.mean(x_errors):.2f}")
    print(f"y_error mean: {np.mean(y_errors):.2f}")
    print(f"z_error mean: {np.mean(z_errors):.2f}")
    print(f"x_error std: {np.std(x_errors):.2f}")
    print(f"y_error std: {np.std(y_errors):.2f}")
    print(f"z_error std: {np.std(z_errors):.2f}")

    # exit()

    # for e, h in zip(x_errors, heights):
    #     data_dict2.append({'axis': 'x', 'value': e, 'lvl': calculate_lvl_from_height(h)})
    # for e, h in zip(y_errors, heights):
    #     data_dict2.append({'axis': 'y', 'value': e, 'lvl': calculate_lvl_from_height(h)})
    # for e, h in zip(z_errors, heights):
    #     data_dict2.append({'axis': 'z', 'value': e, 'lvl': calculate_lvl_from_height(h)})
    for e, h in zip(x_rot_errors, heights):
        data_dict2.append({'axis': 'x', 'value': e, 'height': h})
    for e, h in zip(y_rot_errors, heights):
        data_dict2.append({'axis': 'y', 'value': e, 'height': h})
    for e, h in zip(z_rot_errors, heights):
        data_dict2.append({'axis': 'z', 'value': e, 'height': h})
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


    # fig_dims = (7, 2)
    # fig, ax = plt.subplots(figsize=fig_dims)
    # df = pd.DataFrame(data_dict2)
    #
    # print('Hi!')
    # ax = sns.stripplot(x='lvl', y='value', hue='axis', ax=ax, data=df, dodge=True, size=3)
    # # ax = sns.scatterplot(x='height', y='value', hue='axis', ax=ax, data=df)
    # ax.set(ylabel='Fehler in mm', xlabel='Ebene')
    # plt.legend(title='Achse')
    # plt.tight_layout()
    # plt.show()
    #
    # exit()



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
    dist_norm = stats.norm
    dist_cauchy = stats.cauchy
    dist_hypsecant = stats.hypsecant
    dist_norminv = stats.norminvgauss


    # distributions comparison
    params_norm_x = dist_norm.fit(x_errors)
    params_cauchy_x = dist_cauchy.fit(x_errors)
    params_hypsecant_x = dist_hypsecant.fit(x_errors)
    params_x = dist_norminv.fit(x_errors)
    loglh_norm_x = dist_norm.logpdf(x_errors, params_norm_x[0], params_norm_x[1]).sum()
    loglh_cauchy_x = dist_cauchy.logpdf(x_errors, params_cauchy_x[0], params_cauchy_x[1]).sum()
    loglh_hypsecant_x = dist_hypsecant.logpdf(x_errors, params_hypsecant_x[0], params_hypsecant_x[1]).sum()
    loglh_norminvgauss_x = dist_norminv.logpdf(x_errors, params_x[0], params_x[1], params_x[2], params_x[3]).sum()
    print(f"Loglhs: {loglh_norm_x}, {loglh_cauchy_x}, {loglh_hypsecant_x}, {loglh_norminvgauss_x}")

    params_norm_x = dist_norm.fit(y_errors)
    params_cauchy_x = dist_cauchy.fit(y_errors)
    params_hypsecant_x = dist_hypsecant.fit(y_errors)
    params_x = dist_norminv.fit(y_errors)
    loglh_norm_x = dist_norm.logpdf(y_errors, params_norm_x[0], params_norm_x[1]).sum()
    loglh_cauchy_x = dist_cauchy.logpdf(y_errors, params_cauchy_x[0], params_cauchy_x[1]).sum()
    loglh_hypsecant_x = dist_hypsecant.logpdf(y_errors, params_hypsecant_x[0], params_hypsecant_x[1]).sum()
    loglh_norminvgauss_x = dist_norminv.logpdf(y_errors, params_x[0], params_x[1], params_x[2], params_x[3]).sum()
    print(f"Loglhs: {loglh_norm_x}, {loglh_cauchy_x}, {loglh_hypsecant_x}, {loglh_norminvgauss_x}")

    params_norm_x = dist_norm.fit(z_errors)
    params_cauchy_x = dist_cauchy.fit(z_errors)
    params_hypsecant_x = dist_hypsecant.fit(z_errors)
    params_x = dist_norminv.fit(z_errors)
    loglh_norm_x = dist_norm.logpdf(z_errors, params_norm_x[0], params_norm_x[1]).sum()
    loglh_cauchy_x = dist_cauchy.logpdf(z_errors, params_cauchy_x[0], params_cauchy_x[1]).sum()
    loglh_hypsecant_x = dist_hypsecant.logpdf(z_errors, params_hypsecant_x[0], params_hypsecant_x[1]).sum()
    loglh_norminvgauss_x = dist_norminv.logpdf(z_errors, params_x[0], params_x[1], params_x[2], params_x[3]).sum()
    print(f"Loglhs: {loglh_norm_x}, {loglh_cauchy_x}, {loglh_hypsecant_x}, {loglh_norminvgauss_x}")




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
    min_height = 7.7
    spacing = 14.94
    lvl = int((height - min_height) / spacing)

    return lvl





if __name__ == "__main__":
    evaluate()