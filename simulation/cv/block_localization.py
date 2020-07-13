import cv2
import numpy as np
import dt_apriltags
import math
from utils.utils import *
from .transformations import matrix2pose, getRotationMatrix_180_x, getRotationMatrix_90_z, pose2matrix, get_Ry_h, get_Rx_h, get_Ry, get_Rz_h, get_Rx, get_Rz
from pyquaternion import Quaternion
scaler = 50
one_millimeter = 0.001 * scaler
from constants import detection_threads, quad_decimate
from cv2 import aruco

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_camera_params_mujoco(height, width, fovy):
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    fx = f
    fy = f
    cx = width / 2
    cy = height / 2
    return np.array([fx, fy, cx, cy])


def get_camera_pose_matrix_mujoco(image, tag_id, tag_size, tag_pos, camera_params, detector):
    assert tag_pos.size == 3, "tag_size must be a 3d vector"
    assert len(camera_params) == 4, "camera_params must be a 4d tuple ([fx, fy, sx, sy])"

    detections = detector.detect(image)

    tag_matrix = None

    # get detection with the required apriltag
    for detection in detections:
        if detection.tag_id in [tag_id]:
            R = detection.pose_R
            t = detection.pose_t

    # get camera pos in coordinates of the tag
    if tag_matrix is not None:

        camera_matrix = np.concatenate((R.T, -np.dot(R.T, t)), axis=1)
        last_row = np.reshape([0, 0, 0, 1], (1, 4))
        camera_matrix = np.concatenate((camera_matrix, last_row), axis=0)

        # rotate coordinate frame in order to align with the MuJoCo coordinate system
        camera_matrix = getRotationMatrix_90_z() @ getRotationMatrix_180_x() @ camera_matrix
        camera_pose = matrix2pose(camera_matrix)

        # translate coordinate frame to the MuJoCo coordinate system origin
        camera_pose[0:3] *= tag_size

        # move tag's frame to the MuJoCo origin
        camera_pose[0] += tag_pos[0]
        camera_pose[1] += tag_pos[1]
        camera_pose[2] += tag_pos[2]


        camera_matrix = pose2matrix(camera_pose)

        # return image if flag is set

        return camera_matrix
    else:  # floor tag not detected
        return None


def get_tag_poses_from_image_mujoco(img, target_tag_ids, target_tag_size, ref_tag_id, ref_tag_size, ref_tag_pos, camera_params, detector):
    assert ref_tag_pos.size == 3, "tag_pos_ref must be a 3d vector"

    # get camera pose as a transformation matrix
    camera_pose_matrix = get_camera_pose_matrix_mujoco(img, ref_tag_id, ref_tag_size, ref_tag_pos, camera_params, detector)

    # initialize and start detector
    detections = detector.detect(img, True)

    # return if floor tag not detected
    if camera_pose_matrix is None:
        return None

    # detect block's tag
    block_matrices = dict()
    for detection in detections:
        if detection.tag_id in target_tag_ids:
            R = detection.pose_R
            t = detection.pose_t
            block_matrix = np.concatenate((R, t), axis=1)
            last_row = np.reshape([0, 0, 0, 1], (1, 4))
            block_matrix = np.concatenate((block_matrix, last_row), axis=0)

            block_matrices[detection.tag_id] = block_matrix

    # transform matrices
    for tag_id in block_matrices:
        tag_id = int(tag_id)
        block_matrix = block_matrices[tag_id]

        # scale translation in order to convert into MuJoCo units
        block_matrix[0:3, 3] *= target_tag_size

        # express the target tag position in world coordinates
        block_matrix = camera_pose_matrix @ block_matrix

        # extract frame rotation
        R = block_matrix[0:3, 0:3]

        # for the left tag     for the right tag
        # april     MuJoCo     april     MuJoCo
        #  -x    ~    y          -x    ~    y
        #  -y    ~    z          -y    ~    z
        #   z    ~    x           z    ~    x
        if tag_id % 2 == 0:  # left tag
            R = get_Ry(-90) @ get_Rx(90) @ R
            q = Quaternion(matrix=R)
            q = Quaternion([q.elements[0], q.elements[3], -q.elements[1], -q.elements[2]])
            R = q.rotation_matrix
        else:  # right tag
            R = get_Rx(90) @ get_Rz(-90) @ R
            q = Quaternion(matrix=R)
            q = Quaternion([-q.elements[0], q.elements[3], -q.elements[1], q.elements[2]])
            R = q.rotation_matrix

        block_matrix[:3, :3] = R
        block_matrices[tag_id] = block_matrix


    return block_matrices


def get_block_positions_mujoco(im1, im2, block_ids, target_tag_size, ref_tag_size, ref_tag_id, ref_tag_pos, block_sizes, camera_params, return_images, detector):
    assert block_sizes.size == 3, "block_sizes must be a 3d vector"

    # calculate tag ids corresponding to block ids
    target_tag_ids = []
    for i in block_ids:
        target_tag_ids.append(i * 2)
        target_tag_ids.append(i * 2 + 1)

    tag_pose_matrices_im1, dimage1 = get_tag_poses_from_image_mujoco(img=im1,
                                                                     target_tag_ids=target_tag_ids,
                                                                     target_tag_size=target_tag_size,
                                                                     ref_tag_id=ref_tag_id,
                                                                     ref_tag_size=ref_tag_size,
                                                                     ref_tag_pos=ref_tag_pos,
                                                                     camera_params=camera_params,
                                                                     detector=detector)

    tag_pose_matrices_im2, dimage2 = get_tag_poses_from_image_mujoco(img=im2,
                                                                     target_tag_ids=target_tag_ids,
                                                                     target_tag_size=target_tag_size,
                                                                     ref_tag_id=ref_tag_id,
                                                                     ref_tag_size=ref_tag_size,
                                                                     ref_tag_pos=ref_tag_pos,
                                                                     camera_params=camera_params,
                                                                     detector=detector)
    
    positions = {}

    # if the floor tag war not detected
    if tag_pose_matrices_im1 is None or tag_pose_matrices_im2 is None:
        return positions, dimage1, dimage2
    
    
    for block_id in block_ids:
        left_tag_id = block_id * 2
        right_tag_id = block_id * 2 + 1
    
        # assume that the tag center is on the block's right/left side center
        left_tag_matrix = None
        right_tag_matrix = None
        if left_tag_id in tag_pose_matrices_im1:
            left_tag_matrix = tag_pose_matrices_im1[left_tag_id]
        if left_tag_id in tag_pose_matrices_im2:
            left_tag_matrix = tag_pose_matrices_im2[left_tag_id]
        if right_tag_id in tag_pose_matrices_im1:
            right_tag_matrix = tag_pose_matrices_im1[right_tag_id]
        if right_tag_id in tag_pose_matrices_im2:
            right_tag_matrix = tag_pose_matrices_im2[right_tag_id]

        # if both tags are detected
        if left_tag_matrix is not None and right_tag_matrix is not None:
            right_tag_pose = matrix2pose(right_tag_matrix)
            left_tag_pose = matrix2pose(left_tag_matrix)
            block_center = left_tag_pose[:3] + (right_tag_pose[:3] - left_tag_pose[:3])/2

            # calculate quaternion
            # 1st approach
            # vec = right_tag_pose[:3] - left_tag_pose[:3]
            # rotation_axis = np.cross(np.array([1, 0, 0]), vec)
            # block_quat = Quaternion(axis=rotation_axis, angle=angle_between_vectors(vec, np.array([1, 0, 0])))
            # 2nd approach
            left_tag_quat = Quaternion(matrix=left_tag_matrix)
            right_tag_quat = Quaternion(matrix=right_tag_matrix)
            block_quat = Quaternion(np.mean([left_tag_quat, right_tag_quat]))
            positions[block_id] = {'pos': block_center,
                                   'orientation': block_quat.elements,
                                   'tags_detected': np.array([2])}
        
        elif left_tag_matrix is not None:
            # print(bcolors.WARNING + f"Block #{block_id}: only one tag detected!" + bcolors.ENDC)
            left_tag_pose = matrix2pose(left_tag_matrix)
            tag_quat = Quaternion(matrix=left_tag_matrix)
            tag_normal = tag_quat.rotate(np.array([1, 0, 0]))
            block_center = left_tag_pose[:3] + (tag_normal * block_sizes[0] * 0.5)
            positions[block_id] = {'pos': block_center,
                                   'orientation': tag_quat.elements,
                                   'tags_detected': np.array([1])}
        
        elif right_tag_matrix is not None:
            # print(bcolors.WARNING + f"Block #{block_id}: only one tag detected!" + bcolors.ENDC)
            right_tag_pose = matrix2pose(right_tag_matrix)
            tag_quat = Quaternion(matrix=right_tag_matrix)
            tag_normal = tag_quat.rotate(np.array([1, 0, 0]))
            block_center = right_tag_pose[:3] - (tag_normal * block_sizes[0] * 0.5)
            positions[block_id] = {'pos': block_center,
                                   'orientation': tag_quat.elements,
                                   'tags_detected': np.array([1])}

        else:
            # print(bcolors.FAIL + f"Block #{block_id}: no tags detected!" + bcolors.ENDC)
            pass

    # the last element of each dictionary entry shows how much tags were used for position estimation
    if return_images:
        return positions, dimage1, dimage2
    else:
        return positions

def detect_with_apriltag(img, camera_params, detector, cam_mtx, cam_dist):
    img = cv2.undistort(img, cam_mtx, cam_dist)
    detections = detector.detect(img, True, camera_params, tag_size=1)
    return detections

def detect_with_aruco(img, cam_mtx, cam_dist):
    # initialize aruco
    aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
    aruco_parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=aruco_parameters)

    detections = []
    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 1, cam_mtx, cam_dist)
            rot_mtx, _ = cv2.Rodrigues(rvec)
            tvec = np.reshape(np.array(tvec[0, 0]), (3, 1))

            # transform rot matrix
            q = Quaternion(matrix=rot_mtx)
            q = Quaternion([-q[2], -q[3], q[0], q[1]])
            rot_mtx = q.rotation_matrix

            detection = dt_apriltags.Detection()
            detection.tag_id = ids[i][0]
            detection.pose_t = tvec
            detection.pose_R = rot_mtx

            detections.append(detection)

    return detections


def get_camera_pose_matrix(detections, tag_id, tag_size, tag_pos, camera_params):
    assert tag_pos.size == 3, "tag_size must be a 3d vector"
    assert len(camera_params) == 4, "camera_params must be a 4d tuple ([fx, fy, sx, sy])"

    # print(f"Detections: {detections}")

    R = None
    t = None

    # get detection with the required apriltag
    for detection in detections:
        if detection.tag_id in [tag_id]:
            R = detection.pose_R
            t = detection.pose_t

    # get camera pos in coordinates of the tag
    if R is not None and t is not None:

        camera_matrix = np.concatenate((R.T, -np.dot(R.T, t)), axis=1)
        last_row = np.reshape([0, 0, 0, 1], (1, 4))
        camera_matrix = np.concatenate((camera_matrix, last_row), axis=0)

        # rotate coordinate frame in order to align with the MuJoCo coordinate system
        # camera_matrix = getRotationMatrix_90_z() @ getRotationMatrix_180_x() @ camera_matrix
        camera_pose = matrix2pose(camera_matrix)

        # translate coordinate frame to the MuJoCo coordinate system origin
        camera_pose[0:3] *= tag_size

        # move tag's frame to the MuJoCo origin
        camera_pose[0] += tag_pos[0]
        camera_pose[1] += tag_pos[1]
        camera_pose[2] += tag_pos[2]


        camera_matrix = pose2matrix(camera_pose)

        # return image if flag is set

        return camera_matrix
    else:  # floor tag not detected
        return None


def get_tag_poses_from_image(img, target_tag_ids, target_tag_size, ref_tag_id, ref_tag_size,
                             ref_tag_pos, camera_params, corrections, detector, cam_mtx, cam_dist):
    assert ref_tag_pos.size == 3, "tag_pos_ref must be a 3d vector"

    # detect
    # detections = detector.detect(img, True, camera_params, tag_size=1)
    if detection_method == 'apriltag':
        detections = detect_with_apriltag(img, camera_params, detector, cam_mtx, cam_dist)
    elif detection_method == 'aruco':
        detections = detect_with_aruco(img, cam_mtx, cam_dist)
    else:
        raise ValueError('Wrong detection method!')


    # get camera pose as a transformation matrix
    camera_pose_matrix = get_camera_pose_matrix(detections, ref_tag_id, ref_tag_size, ref_tag_pos, camera_params)

    # return if floor tag not detected
    if camera_pose_matrix is None:
        return None

    # detect block's tag
    block_matrices = dict()
    for detection in detections:
        if detection.tag_id in target_tag_ids:
            R = detection.pose_R
            t = detection.pose_t
            block_matrix = np.concatenate((R, t), axis=1)
            last_row = np.reshape([0, 0, 0, 1], (1, 4))
            block_matrix = np.concatenate((block_matrix, last_row), axis=0)
            block_matrices[detection.tag_id] = block_matrix

    # transform matrices
    for tag_id in block_matrices:
        tag_id = int(tag_id)
        block_matrix = block_matrices[tag_id]

        # scale translation in order to convert into MuJoCo units
        block_matrix[0:3, 3] *= target_tag_size

        # express the target tag position in world coordinates
        block_matrix = camera_pose_matrix @ block_matrix

        # extract frame rotation
        R = block_matrix[0:3, 0:3]

        # for the left tag     for the right tag
        # april     MuJoCo     april     MuJoCo
        #  -x    ~    y          -x    ~    y
        #  -y    ~    z          -y    ~    z
        #   z    ~    x           z    ~    x
        # if tag_id % 2 == 0:  # left tag
        #     R = get_Ry(-90) @ get_Rx(90) @ R
        #     q = Quaternion(matrix=R)
        #     q = Quaternion([q.elements[0], q.elements[3], -q.elements[1], -q.elements[2]])
        #     R = q.rotation_matrix
        # else:  # right tag
        #     R = get_Rx(90) @ get_Rz(-90) @ R
        #     q = Quaternion(matrix=R)
        #     q = Quaternion([-q.elements[0], q.elements[3], -q.elements[1], q.elements[2]])
        #     R = q.rotation_matrix

        block_matrix[:3, :3] = R

        # convert rot matrix to quaternion for further correction
        q = Quaternion(matrix=R)

        # correct rotation
        q = correct_quat(q, corrections[tag_id]['quat_corr'])

        # correct pos
        t = block_matrix[0:3, 3]
        t = np.reshape(t, (1, 3))
        t = correct_position(t, corrections[tag_id]['pos_corr'], q)
        t = np.reshape(t, (3, 1))

        # convert quaternion back to matrix
        R = q.rotation_matrix

        block_matrix = np.concatenate((R, t), axis=1)
        last_row = np.reshape([0, 0, 0, 1], (1, 4))
        block_matrix = np.concatenate((block_matrix, last_row), axis=0)



        block_matrices[tag_id] = block_matrix


    return block_matrices


def get_block_positions(im1, im2, block_ids, target_tag_size, ref_tag_size, ref_tag_id, ref_tag_pos, block_sizes,
                        corrections, camera_params1, camera_params2, return_images, detector, cam_mtx1, cam_dist1, cam_mtx2, cam_dist2):

    # calculate tag ids corresponding to block ids
    target_tag_ids = []
    for i in block_ids:
        target_tag_ids.append(i * 2)
        target_tag_ids.append(i * 2 + 1)

    tag_pose_matrices_im1 = get_tag_poses_from_image(img=im1,
                                                     target_tag_ids=target_tag_ids,
                                                     target_tag_size=target_tag_size,
                                                     ref_tag_id=ref_tag_id,
                                                     ref_tag_size=ref_tag_size,
                                                     ref_tag_pos=ref_tag_pos,
                                                     camera_params=camera_params1,
                                                     corrections=corrections,
                                                     detector=detector,
                                                     cam_mtx=cam_mtx1,
                                                     cam_dist=cam_dist1)

    tag_pose_matrices_im2 = get_tag_poses_from_image(img=im2,
                                                     target_tag_ids=target_tag_ids,
                                                     target_tag_size=target_tag_size,
                                                     ref_tag_id=ref_tag_id,
                                                     ref_tag_size=ref_tag_size,
                                                     ref_tag_pos=ref_tag_pos,
                                                     camera_params=camera_params2,
                                                     corrections=corrections,
                                                     detector=detector,
                                                     cam_mtx=cam_mtx2,
                                                     cam_dist=cam_dist2)

    positions = {}

    # if the floor tag war not detected
    if tag_pose_matrices_im1 is None and tag_pose_matrices_im2 is None:
        return positions

    for block_id in block_ids:
        left_tag_id = block_id * 2
        right_tag_id = block_id * 2 + 1

        # assume that the tag center is on the block's right/left side center
        left_tag_matrix = None
        right_tag_matrix = None

        if tag_pose_matrices_im1 is not None:
            if left_tag_id in tag_pose_matrices_im1:
                left_tag_matrix = tag_pose_matrices_im1[left_tag_id]
            if right_tag_id in tag_pose_matrices_im1:
                right_tag_matrix = tag_pose_matrices_im1[right_tag_id]

        if tag_pose_matrices_im2 is not None:
            if left_tag_id in tag_pose_matrices_im2:
                left_tag_matrix = tag_pose_matrices_im2[left_tag_id]
            if right_tag_id in tag_pose_matrices_im2:
                right_tag_matrix = tag_pose_matrices_im2[right_tag_id]

        # if both tags are detected
        if left_tag_matrix is not None and right_tag_matrix is not None:
            right_tag_pose = matrix2pose(right_tag_matrix)
            left_tag_pose = matrix2pose(left_tag_matrix)
            block_center = left_tag_pose[:3] + (right_tag_pose[:3] - left_tag_pose[:3]) / 2

            # calculate quaternion
            # 1st approach
            # vec = right_tag_pose[:3] - left_tag_pose[:3]
            # rotation_axis = np.cross(np.array([1, 0, 0]), vec)
            # block_quat = Quaternion(axis=rotation_axis, angle=angle_between_vectors(vec, np.array([1, 0, 0])))
            # 2nd approach
            left_tag_quat = quat_canonical_form(Quaternion(matrix=left_tag_matrix))
            print(f'Left tag quat: {left_tag_quat}')
            right_tag_quat = quat_canonical_form(Quaternion(matrix=right_tag_matrix) * Quaternion(axis=[0, 1, 0], degrees=180))
            print(f'Right tag quat: {right_tag_quat}')
            block_quat = Quaternion(np.mean([left_tag_quat, right_tag_quat]))
            positions[block_id] = {'pos': block_center,
                                   'orientation': block_quat.elements,
                                   'tags_detected': np.array([2])}

        elif left_tag_matrix is not None or right_tag_matrix is not None:
            # print(bcolors.WARNING + f"Block #{block_id}: only one tag detected!" + bcolors.ENDC)
            if left_tag_matrix is not None:
                tag_matrix = left_tag_matrix
            else:
                tag_matrix = right_tag_matrix

            tag_pose = matrix2pose(tag_matrix)
            tag_quat = Quaternion(matrix=tag_matrix)
            tag_normal = tag_quat.rotate(np.array([0, 0, 1]))
            block_center = tag_pose[:3] + (tag_normal * block_sizes[block_id]['length'] * 0.5)
            if right_tag_matrix is not None:
                block_quat = tag_quat * Quaternion(axis=[0, 1, 0], degrees=180)
            else:
                block_quat = tag_quat
            positions[block_id] = {'pos': block_center,
                                   'orientation': block_quat,
                                   'tags_detected': np.array([1])}

        # elif right_tag_matrix is not None:
        #     # print(bcolors.WARNING + f"Block #{block_id}: only one tag detected!" + bcolors.ENDC)
        #     right_tag_pose = matrix2pose(right_tag_matrix)
        #     tag_quat = Quaternion(matrix=right_tag_matrix)
        #     tag_normal = tag_quat.rotate(np.array([1, 0, 0]))
        #     block_center = right_tag_pose[:3] - (tag_normal * block_sizes[0] * 0.5)
        #     positions[block_id] = {'pos': block_center,
        #                            'orientation': tag_quat.elements,
        #                            'tags_detected': np.array([1])}

        else:
            # print(bcolors.FAIL + f"Block #{block_id}: no tags detected!" + bcolors.ENDC)
            pass

    # the last element of each dictionary entry shows how much tags were used for position estimation
    if return_images:
        return positions
    else:
        return positions

def correct_position(current_pos, correction, quat):
    print(f"Current pos: {current_pos}")
    print(f"Correction: {correction}")
    res = current_pos
    res += correction[0] * quat.rotate(x_unit_vector)
    res += correction[1] * quat.rotate(y_unit_vector)
    res += correction[2] * quat.rotate(z_unit_vector)

    return res

def correct_quat(current_quat, corrective_quat):
    return current_quat * corrective_quat


if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True)

    im1 = cv2.imread('/home/bch_svt/cartpole/simulation/screenshots/im1.png', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('/home/bch_svt/cartpole/simulation/screenshots/im2.png', cv2.IMREAD_GRAYSCALE)

    height, width = im1.shape
    camera_params = np.array(get_camera_params_mujoco(height, width, 45))
    detector = dt_apriltags.Detector(nthreads=detection_threads, quad_decimate=quad_decimate)
    pose = get_block_positions_mujoco(im1=im1,
                                      im2=im2,
                                      block_ids=[3],
                                      target_tag_size=0.015 * scaler * 0.8,
                                      ref_tag_size=0.06 * scaler * 0.8,
                                      ref_tag_id=255,
                                      ref_tag_pos=scaler*np.array([-0.1, -0.1, 0]),
                                      block_sizes=scaler * np.array([0.075, 0.025, 0.015]),
                                      camera_params=camera_params,
                                      detector=detector)

    print(pose[3][:3]/one_millimeter)
    print(pose[3][3:])
    print(pose)