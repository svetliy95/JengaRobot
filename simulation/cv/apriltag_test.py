import apriltag
import dt_apriltags
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
import math

def get_camera_params_mujoco(height, width, fovy):
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    fx = f
    fy = f
    cx = width / 2
    cy = height / 2
    return (fx, fy, cx, cy)


if __name__ == "__main__":

    detection_threads = 16
    quad_decimate = 1
    quad_sigma = 1.18


    # detector #1
    detector1_optionen = apriltag.DetectorOptions(nthreads=detection_threads, quad_decimate=quad_decimate, debug=True)
    detector1 = apriltag.Detector(detector1_optionen)

    # detector #2
    detector2 = dt_apriltags.Detector(nthreads=detection_threads, quad_decimate=quad_decimate, quad_sigma=quad_sigma)

    # test image
    # im = cv2.imread('/home/bch_svt/cartpole/simulation/screenshots/screenshot.png', cv2.IMREAD_GRAYSCALE)
    # im = cv2.imread('/home/bch_svt/cartpole/simulation/cv/pictures/test_image.bmp', cv2.IMREAD_GRAYSCALE)
    im = cv2.imread('/home/bch_svt/cartpole/simulation/cv/test_image_cali.bmp', cv2.IMREAD_GRAYSCALE)

    # detect #1
    detections1 = detector1.detect(im)
    print(f"Detections #1: {len(detections1)}")

    # detect #2
    detections2 = detector2.detect(im)
    print(f"Detections #2: {len(detections2)}")

    # declare stuff for ploting
    sharps = np.arange(1, 5, 0.1)
    sigmas = np.arange(1.1, 1.7, 0.1)
    area = []
    x = []
    y = []
    max = 0

    height, width = im.shape
    camera_params_mujoco = get_camera_params_mujoco(height, width, 45)

    # test different parameters
    for decode_sharpening in sharps:
        for quad_sigma in sigmas:
            detector2 = dt_apriltags.Detector(nthreads=detection_threads, quad_decimate=quad_decimate,
                                              quad_sigma=quad_sigma, decode_sharpening=decode_sharpening)
            # detect #2
            detections2 = detector2.detect(im, True, camera_params_mujoco, tag_size=1)
            # detections2 = detector2.detect(im)
            # print(detections2)
            l = len(detections2)
            print(f"Detections (sigma={quad_sigma:.2f}, shrpening={decode_sharpening:.2f}): {l}")

            x.append(decode_sharpening)
            y.append(quad_sigma)
            area.append(l)
            if l > max:
                best_params = (quad_sigma, decode_sharpening)
                max = l


    print(f"Max: {max}")
    print(f"Best params: {best_params}")
    plt.scatter(x, y, c=area)
    plt.show()






