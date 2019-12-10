import cv2
import numpy as np
import apriltag
import time
import math
from transformations import matrix2pose, getRotationMatrix_180_x, getRotationMatrix_90_z, pose2matrix, getRotationMatrix_90_x, get_Rx, get_Ry, get_Rz
from pyquaternion import Quaternion
from PIL import Image, ImageDraw
scaler = 50
one_millimeter = 0.001 * scaler

def cameraPoseFromHomography(H):
    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = np.cross(H1, H2)

    norm1 = np.linalg.norm(H1)
    norm2 = np.linalg.norm(H2)
    tnorm = (norm1 + norm2) / 2.0

    T = H[:, 2] / tnorm
    return np.mat([H1, H2, H3, T])


def get_camera_params(im_height, fovy):
    f = 0.5 * im_height / math.tan(fovy * math.pi / 360)
    fx = f
    fy = f
    sx = width / 2
    sy = height / 2
    return [fx, fy, sx, sy]


def get_camera_pose_matrix(image, tag_id, tag_size, tag_pos, camera_params):
    assert tag_pos.size == 3, "tag_size must be a 3d vector"
    assert camera_params.size == 4, "camera_params must be a 4d vector ([fx, fy, sx, sy])"
    height, width = image.shape


    detector = apriltag.Detector()
    detections = detector.detect(image, False)

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
        print("Tag pos: " + str(tag_pos))
        actual_tag_pos = tag_pos - np.array([0, 0, 0])  # middle point of the tag's left side (tag's frame origin)
        print("Actual tag pos: " + str(actual_tag_pos))
        camera_pose[0:3] *= tag_size

        # move tag's frame to the MuJoCo origin
        camera_pose[0] += actual_tag_pos[0]
        camera_pose[1] += actual_tag_pos[1]
        camera_pose[2] += tag_pos[2]


        camera_matrix = pose2matrix(camera_pose)

        return camera_matrix
    else:
        return None


def get_tag_pose_from_image(img, target_tag_id, target_tag_size, ref_tag_id, ref_tag_size, ref_tag_pos, block_sizes, camera_params):
    assert ref_tag_pos.size == 3, "tag_pos_ref must be a 3d vector"
    assert block_sizes.size == 3, "block_sizes must be a 3d vector"

    # get camera pose as a transformation matrix
    camera_pose_matrix = get_camera_pose_matrix(img, ref_tag_id, ref_tag_size, ref_tag_pos, camera_params)

    # initialize and start detector
    detector = apriltag.Detector()
    detections, image = detector.detect(img, True)

    # detect block's tag
    block_matrix = None
    for detection in detections:
        if detection.tag_id is target_tag_id:
            block_matrix, _, _ = detector.detection_pose(detection, camera_params)
            break

    # return None if the tag could not be detected
    if block_matrix is None:
        return None

    # scale translation in order to convert into MuJoCo units
    block_matrix[0:3, 3] *= target_tag_size

    # express the target tag position in world coordinates
    block_matrix = camera_pose_matrix @ block_matrix

    # rotate coordinate frame to align with the MuJoCo global coordinate system
    block_matrix = getRotationMatrix_90_z() @ getRotationMatrix_90_x() @ block_matrix

    return block_matrix


imagepath = '/home/bch_svt/cartpole/simulation/screenshots/screenshot.png'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
height, width = image.shape

camera_params = np.array(get_camera_params(height, 45))
block_pos = get_tag_pose_from_image(img=image,
                                    target_tag_id=1,
                                    target_tag_size=0.015*scaler*0.8,
                                    ref_tag_id=255,
                                    ref_tag_size=0.1*scaler*0.8,
                                    ref_tag_pos=scaler * np.array([-0.1, -0.1, 0]),
                                    block_sizes=scaler * np.array([0.075, 0.025, 0.015]),
                                    camera_params=camera_params)

print(matrix2pose(block_pos))

camera_matrix = get_camera_pose_matrix(image, 255, 0.06 * scaler * 0.8, scaler * np.array([-0.1, -0.1, 0]), camera_params)

camera_pose = matrix2pose(camera_matrix)

print(np.array_str(camera_pose, precision=3, suppress_small=True))

detector = apriltag.Detector()
detections, image = detector.detect(image, True)

cv2.imshow("pic", image)
cv2.waitKey(100000)

# camera params
fx, fy, sx, sy = get_camera_params(height, 45)
body_matrix = None
corners = None
for detection in detections:
    if detection.tag_id in [1]:
        body_matrix, _, _ = detector.detection_pose(detection, [fx, fy, sx, sy])
        corners = detection.corners
        print([detection])

image = Image.fromarray(image)
draw = ImageDraw.Draw(image)
print(np.array([corners[0][0], corners[0][1], corners[1][0], corners[1][1]]))
draw.rectangle([corners[0][0], corners[0][1], corners[1][0], corners[1][1]], 128, 128, 1)

image.show()


# body_matrix[0:3, 3] *= 0.1*scaler*0.8
body_matrix[0:3, 3] *= 0.0148 * scaler * 0.8

print(body_matrix)
body_matrix = camera_matrix @ body_matrix
body_matrix[0:3, 3] /= one_millimeter
print(np.array_str(matrix2pose(body_matrix), precision=3, suppress_small=True))


q = Quaternion(matrix=body_matrix)

print(q.yaw_pitch_roll)