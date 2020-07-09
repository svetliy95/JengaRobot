from cv2 import aruco
import dt_apriltags
import time
from constants import *
import cv2
from cv.camera import Camera

def detect_with_apriltag(im, cam_params, detector):
    detections = detector.detect(im, True, cam_params, tag_size=1)
    # detections2 = detector.detect(im2, True, cam2_params, tag_size=1)

    return detections

def detect_with_aruco(im, aruco_dict, params):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(im, aruco_dict, parameters=params)

    poses = dict()
    print(ids)
    for i in range(len(ids)):
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 1, cam1_mtx, cam1_dist)
        rot_mtx, _ = cv2.Rodrigues(rvec)
        tvec = np.array(tvec[0, 0]).transpose()
        poses[ids[i][0]] = {'rot_mtx': rot_mtx, 't': tvec}


    print(poses)

    return poses

def compare_apriltag_and_aruco(im, at_detector):
    pass

if __name__ == "__main__":
    cam1 = Camera(cam1_serial, cam1_mtx, cam1_dist)
    cam2 = Camera(cam2_serial, cam2_mtx, cam2_dist)
    cam1_params = cam1.get_params()
    cam2_params = cam2.get_params()
    cam_params = cam1.get_params()
    quad_decimate = 1
    at_detector = dt_apriltags.Detector(nthreads=detection_threads,
                                     quad_decimate=quad_decimate,
                                     quad_sigma=quad_sigma,
                                     decode_sharpening=decode_sharpening)
    im1 = cv2.imread('./pictures/cam1_test_image.bmp', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('./pictures/cam2_test_image.bmp', cv2.IMREAD_GRAYSCALE)
    im = cv2.imread('./pictures/cam1_test_image3.bmp', cv2.IMREAD_GRAYSCALE)

    # initialize aruco
    aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
    aruco_parameters = aruco.DetectorParameters_create()
    aruco_parameters.aprilTagQuadDecimate = 1000.0
    aruco_parameters.aprilTagQuadSigma = 1.3
    print(f"Quad decimate: {aruco_parameters.aprilTagQuadDecimate}")
    print(f"Quad sigma: {aruco_parameters.aprilTagQuadSigma}")
    # aruco_parameters.minMarkerPerimeterRate = 0.01
    # aruco_parameters.maxMarkerPerimeterRate = 0.9

    # test apriltag
    start = time.time()
    det = detect_with_apriltag(im, cam_params, at_detector)
    elapsed = time.time() - start
    print(f"Apriltag. Detections: {det}")
    print(f'Elapsed time: {elapsed*1000:.2f}ms')

    # test aruco
    start = time.time()
    det = detect_with_aruco(im, aruco_dict, aruco_parameters)
    elapsed = time.time() - start
    print(f"ArUco. Detections: {det}")
    print(f'Elapsed time: {elapsed * 1000:.2f}ms')

