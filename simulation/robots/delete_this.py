from mpl_toolkits import mplot3d
from pyquaternion import Quaternion

import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
import seaborn as sns
import time
import pandas as pd

# fig = plt.figure()
# ax = plt.axes(projection="3d")
#
# z_line = np.linspace(0, 15, 1000)
# x_line = np.cos(z_line)
# y_line = np.sin(z_line)
# ax.plot3D(x_line, y_line, z_line, 'gray')
#
# z_points = 15 * np.random.random(100)
# x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
# y_points = np.sin(z_points) + 0.1 * np.random.randn(100)
# ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')


def _get_quarter(pos, tower_center):
    assert pos.size == 3, "Argument 'pos' is not a 3d array!"
    x = pos[0] - tower_center[0]
    y = pos[1] - tower_center[1]
    if y >= abs(x):
        return 1
    if y <= -abs(x):
        return 3
    if x >= 0 and abs(y) < abs(x):
        return 4
    if x < 0 and abs(y) < abs(x):
        return 2

# plt.show()

if __name__ == "__main__":
    xs = np.random.random(1000) * 10
    ys = np.random.random(1000) * 10
    data = [{'x': x, 'y': y, 'quarter': _get_quarter(np.array([x, y, 0]), np.array([5, 5, 0])) } for x, y in zip(xs, ys)]
    data = pd.DataFrame(data)
    sns.scatterplot(x='x', y='y', hue='quarter', data=data)
    plt.show()
