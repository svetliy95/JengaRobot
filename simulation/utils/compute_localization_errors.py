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

def evaluate():
    folder = '/home/bch_svt/cartpole/simulation/localization_errors_with_quaternions'
    filename_prefix = 'localization_errors*'
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
    for f in file_names:

        with open(folder + '/' + f) as file:
            print(file.name)
            reader = csv.reader(file)
            data = list(reader)
            for row in data:
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

    # concert in degrees
    x_rot_errors = list(map(math.degrees, x_rot_errors + list(map(lambda x: -x, x_rot_errors))))
    y_rot_errors = list(map(math.degrees, y_rot_errors + list(map(lambda x: -x, y_rot_errors))))
    z_rot_errors = list(map(math.degrees, z_rot_errors + list(map(lambda x: -x, z_rot_errors))))

    # remove outliers
    x_errors = remove_outliers(x_errors)
    y_errors = remove_outliers(y_errors)
    z_errors = remove_outliers(z_errors)
    a = remove_outliers(a)
    b = remove_outliers(b)
    c = remove_outliers(c)
    d = remove_outliers(d)

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
    errors = d


    sns.distplot(errors, bins=500)
    # sns.scatterplot(x_errors, b)

    # compute parameters for all features
    # dist = stats.norm
    # dist = stats.cauchy
    # dist = stats.hypsecant
    dist = stats.norminvgauss
    params_x = dist.fit(x_errors)
    params_y = dist.fit(y_errors)
    params_z = dist.fit(z_errors)
    params_a = dist.fit(a)
    params_b = dist.fit(b)
    params_c = dist.fit(c)
    params_d = dist.fit(d)
    print(f"Params x: {params_x}")
    print(f"Params y: {params_y}")
    print(f"Params z: {params_z}")
    print(f"Params a: {params_a}")
    print(f"Params b: {params_b}")
    print(f"Params c: {params_c}")
    print(f"Params d: {params_d}")

    params = params_d

    mu = np.mean(errors)
    sigma = np.std(errors)
    print(f"Mean: {mu:.2f}, std: {sigma:.2f}")

    # compute log likelihood of the fitted dist
    loglh = dist.logpdf(errors, params[0], params[1], params[2], params[3]).sum()
    print(f"Log likelihood: {loglh}")

    # compute fitted dist
    x = np.linspace(-3*sigma, 3*sigma, 100)
    plt.plot(x, dist.pdf(x, params[0], params[1], params[2], params[3]))
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # plot fitted dist
    # plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()





if __name__ == "__main__":
    evaluate()