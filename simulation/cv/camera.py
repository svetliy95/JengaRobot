import cv2
import numpy as np
import os
import glob
from pypylon import genicam
from pypylon import pylon
import logging
import colorlog
from constants import *

# initialize logging
log = logging.Logger(__name__)
formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)sPID:%(process)d:%(funcName)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
log.addHandler(stream_handler)

class Camera:
    def __init__(self, serial, mtx, dist):
        """

        :param serial: camera serial number
        :param mtx: camera matrix
        :param dist: distortion coefficients for image rectification
        """

        # initialise camera matrix and dist coefficients
        self.mtx = mtx
        self.dist = dist

        # get available devices
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        # find camera with given serial
        device = None
        for d in devices:
            if d.GetSerialNumber() == str(serial):
                device = d

        if device == None:
            log.error(f"Cannot find the device with the following serial number: {serial}!")
            return
        device_instace = tlFactory.CreateDevice(device)
        self.camera = pylon.InstantCamera(tlFactory.CreateDevice(device))
        self.camera.Open()

    @staticmethod
    def calibrate(fnames):

        # Defining the dimensions of checkerboard
        CHECKERBOARD = (12, 23)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        # Extracting path of individual image stored in a given directory
        # images = glob.glob('./camera_calibration_images/*.jpg')
        for fname in fnames:
            print(f"Current image: {fname}")
            img = cv2.imread(fname)
            print(img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                print(f"Checkerboard found!")
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            else:
                print(f"Checkerboard not found!")

            # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('img', 1200, 1200)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

        cv2.destroyAllWindows()

        h, w = img.shape[:2]

        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        tot_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error
            print(f"Current error: {error}")

        mean_error = tot_error / len(objpoints)

        print (f"Mean error: {mean_error}")

        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        print("rvecs : \n")
        print(rvecs)
        print("tvecs : \n")
        print(tvecs)

        return mtx, dist

    def get_params(self):
        return self.mtx[0][0], self.mtx[1][1], self.mtx[0][2], self.mtx[1][2]

    @staticmethod
    def _scale_image(img, ratio):
        width = int(img.shape[1] * ratio)
        height = int(img.shape[0] * ratio)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim)
        # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist)

    def take_picture(self, scaler=1):
        image = self.camera.GrabOne(1000).Array
        undistorted_image = cv2.undistort(image, self.mtx, self.dist)
        # image = np.rot90(image, 3)
        # undistorted_image = np.rot90(undistorted_image, 3)

        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img', 1200, 1200)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # cv2.imshow('img', image - undistorted_image)
        # cv2.waitKey(0)
        if scaler != 1:
            undistorted_image = self._scale_image(undistorted_image, scaler)
        return undistorted_image

    def take_raw_picture(self):
        image = self.camera.GrabOne(1000).Array
        # image = np.rot90(image, 3)
        # undistorted_image = np.rot90(undistorted_image, 3)

        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img', 1200, 1200)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # cv2.imshow('img', image - undistorted_image)
        # cv2.waitKey(0)
        return image

if __name__ == '__main__':
    fnames = glob.glob('./cam2_calibration/*.bmp')
    mtx, dist = Camera.calibrate(fnames)
    # mtx_diff = np.divide(abs(mtx-cam1_mtx_11cm_2), mtx) * 100
    # dist_diff = np.divide(abs(dist-cam1_dist_11cm_2), dist) * 100
    # print(mtx_diff)
    # print(dist_diff)
    print(repr(mtx))
    print(repr(dist))
    # cam = Camera('22917552', mtx, dist)
    # im = cam.take_picture()
    # print(f"Shape: {im.shape}")

