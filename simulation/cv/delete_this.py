import dt_apriltags
from constants import *
import cv2
from cv.camera import Camera
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats


if __name__ == "__main__":
    dist = stats.norminvgauss(x_error_dist_params)
