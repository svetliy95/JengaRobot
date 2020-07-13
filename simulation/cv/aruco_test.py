from cv2 import aruco
import dt_apriltags
import time
from constants import *
import cv2
from cv.camera import Camera
from utils.utils import mtx_diff
from pyquaternion import Quaternion

def detect_with_apriltag(im, cam_params, detector):
    detections = detector.detect(im, True, cam_params, tag_size=1)
    poses = dict()

    for det in detections:
        rot_mtx = det.pose_R
        t = det.pose_t
        t_mtx = np.concatenate((rot_mtx, t), axis=1)
        last_row = np.reshape(np.array([0, 0, 0, 1]), (1, 4))
        t_mtx = np.concatenate((t_mtx, last_row), axis=0)
        poses[det.tag_id] = {'rot_mtx': rot_mtx, 't': t, 't_mtx': t_mtx}

    return poses

def detect_with_aruco(im, aruco_dict, params):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(im, aruco_dict, parameters=params)

    poses = dict()
    print(ids)
    for i in range(len(ids)):
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 1, cam1_mtx, cam1_dist)
        rot_mtx, _ = cv2.Rodrigues(rvec)
        tvec = np.reshape(np.array(tvec[0, 0]), (3, 1))
        t_mtx = np.concatenate((rot_mtx, tvec), axis=1)
        last_row = np.reshape(np.array([0, 0, 0, 1]), (1, 4))
        t_mtx = np.concatenate((t_mtx, last_row), axis=0)
        poses[ids[i][0]] = {'rot_mtx': rot_mtx, 't': tvec, 't_mtx': t_mtx}

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
    at_detector_reduced = dt_apriltags.Detector(nthreads=detection_threads,
                                        quad_decimate=2,
                                        quad_sigma=quad_sigma,
                                        decode_sharpening=decode_sharpening)
    im1 = cv2.imread('./pictures/cam1_test_image.bmp', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('./pictures/cam2_test_image.bmp', cv2.IMREAD_GRAYSCALE)
    im = cv2.imread('./pictures/cam1_test_image3.bmp', cv2.IMREAD_GRAYSCALE)

    # initialize aruco
    aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
    aruco_parameters = aruco.DetectorParameters_create()
    # aruco_parameters.minMarkerPerimeterRate = 0.01
    # aruco_parameters.maxMarkerPerimeterRate = 0.9

    # test apriltag
    start = time.time()
    det_ap = detect_with_apriltag(cam1.undistort(im), cam_params, at_detector)
    elapsed = time.time() - start
    # print(f"Apriltag. Detections: {det}")
    print(f'Elapsed time: {elapsed*1000:.2f}ms')

    # test apriltag reduced
    start = time.time()
    det_ap_reduced = detect_with_apriltag(cam1.undistort(im), cam_params, at_detector_reduced)
    elapsed = time.time() - start
    print(f'Elapsed time: {elapsed * 1000:.2f}ms')

    # test aruco
    start = time.time()
    det_aruco = detect_with_aruco(im, aruco_dict, aruco_parameters)
    elapsed = time.time() - start
    # print(f"ArUco. Detections: {det}")
    print(f'Elapsed time: {elapsed * 1000:.2f}ms')

    # compare results
    np.set_printoptions(precision=3, suppress=True)
    for id in det_ap:
        if id in det_aruco:

            # AprilTag
            print(f"AprilTag mtx:")
            mtx_at = det_ap[id]['t_mtx']
            print(mtx_at)
            q_at = Quaternion(matrix=mtx_at)

            # AprilTag reduced
            print(f"AprilTag mtx reduced:")
            mtx_at_reduced = det_ap_reduced[id]['t_mtx']
            print(mtx_at_reduced)
            q_at_reduced = Quaternion(matrix=mtx_at_reduced)

            #ArUco
            mtx_aruco = det_aruco[id]['t_mtx']
            q_aruco = Quaternion(matrix=mtx_aruco)

            # transform matrix
            q_aruco_new = Quaternion([-q_aruco[2], -q_aruco[3], q_aruco[0], q_aruco[1]])
            mtx_aruco_new = q_aruco_new.transformation_matrix
            mtx_aruco_new[0:3, 3] = mtx_aruco[0:3, 3]
            print(f"New transformation matrix:")
            print(mtx_aruco_new)
            print(f"New diff:")
            print(mtx_diff(mtx_at, mtx_aruco_new))


            # difference between reduced and full
            print('Diff between reduced and full:')
            print(mtx_diff(mtx_at, mtx_at_reduced))


