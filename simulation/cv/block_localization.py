import cv2
import numpy as np
import apriltag
import math
from utils import *
from .transformations import matrix2pose, getRotationMatrix_180_x, getRotationMatrix_90_z, pose2matrix, get_Ry_h, get_Rx_h, get_Ry, get_Rz_h, get_Rx, get_Rz
from pyquaternion import Quaternion
scaler = 50
one_millimeter = 0.001 * scaler

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_camera_params(height, width, fovy):
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    fx = f
    fy = f
    sx = width / 2
    sy = height / 2
    return np.array([fx, fy, sx, sy])


def get_camera_pose_matrix(image, tag_id, tag_size, tag_pos, camera_params):
    assert tag_pos.size == 3, "tag_size must be a 3d vector"
    assert camera_params.size == 4, "camera_params must be a 4d vector ([fx, fy, sx, sy])"

    detector = apriltag.Detector()
    detections = detector.detect(image)

    tag_matrix = None

    # get detection with the required apriltag
    for detection in detections:
        if detection.tag_id in [tag_id]:
            tag_matrix, _, _ = detector.detection_pose(detection, camera_params)

    # get camera pos in coordinates of the tag
    if tag_matrix is not None:
        R = tag_matrix[0:3, 0:3]
        t = np.reshape(tag_matrix[:-1, -1], (3, 1))

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


def get_tag_poses_from_image(img, target_tag_ids, target_tag_size, ref_tag_id, ref_tag_size, ref_tag_pos, camera_params):
    assert ref_tag_pos.size == 3, "tag_pos_ref must be a 3d vector"

    # get camera pose as a transformation matrix
    camera_pose_matrix = get_camera_pose_matrix(img, ref_tag_id, ref_tag_size, ref_tag_pos, camera_params)

    # initialize and start detector
    detector = apriltag.Detector()
    detections, dimage = detector.detect(img, True)

    # return if floor tag not detected
    if camera_pose_matrix is None:
        return None, dimage

    # detect block's tag
    block_matrices = dict()
    for detection in detections:
        if detection.tag_id in target_tag_ids:
            block_matrix, _, _ = detector.detection_pose(detection, camera_params)
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


    return block_matrices, dimage

def get_block_positions(im1, im2, block_ids, target_tag_size, ref_tag_size, ref_tag_id, ref_tag_pos, block_sizes, camera_params, return_images):
    assert block_sizes.size == 3, "block_sizes must be a 3d vector"

    # calculate tag ids corresponding to block ids
    target_tag_ids = []
    for i in block_ids:
        target_tag_ids.append(i * 2)
        target_tag_ids.append(i * 2 + 1)

    tag_pose_matrices_im1, dimage1 = get_tag_poses_from_image(img=im1,
                                                                target_tag_ids=target_tag_ids,
                                                                target_tag_size=target_tag_size,
                                                                ref_tag_id=ref_tag_id,
                                                                ref_tag_size=ref_tag_size,
                                                                ref_tag_pos=ref_tag_pos,
                                                                camera_params=camera_params)

    tag_pose_matrices_im2, dimage2 = get_tag_poses_from_image(img=im2,
                                                     target_tag_ids=target_tag_ids,
                                                     target_tag_size=target_tag_size,
                                                     ref_tag_id=ref_tag_id,
                                                     ref_tag_size=ref_tag_size,
                                                     ref_tag_pos=ref_tag_pos,
                                                     camera_params=camera_params)
    
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


if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True)

    im1 = cv2.imread('/home/bch_svt/cartpole/simulation/screenshots/im1.png', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('/home/bch_svt/cartpole/simulation/screenshots/im2.png', cv2.IMREAD_GRAYSCALE)

    height, width = im1.shape
    camera_params = np.array(get_camera_params(height, width, 45))

    pose = get_block_positions(im1=im1,
                               im2=im2,
                               block_ids=[3],
                               target_tag_size=0.015 * scaler * 0.8,
                               ref_tag_size=0.06 * scaler * 0.8,
                               ref_tag_id=255,
                               ref_tag_pos=scaler*np.array([-0.1, -0.1, 0]),
                               block_sizes=scaler * np.array([0.075, 0.025, 0.015]),
                               camera_params=camera_params)

    print(pose[3][:3]/one_millimeter)
    print(pose[3][3:])
    print(pose)