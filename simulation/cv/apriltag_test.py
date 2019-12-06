import cv2
import numpy as np
import apriltag
import time
import math
from transformations import matrix2pose, getRotationMatrix_180_x, getRotationMatrix_90_z, pose2matrix
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

def get_camera_pose(image, tag_id, tag_size, tag_pos, tag_modules_in_row=8):
    imagepath = '/home/bch_svt/cartpole/simulation/screenshots/screenshot.png'
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape


    detector = apriltag.Detector()
    detections = detector.detect(image, False)

    # camera params
    params = get_camera_params(height, 45)

    tag_matrix = None

    # get detection with the required apriltag
    for detection in detections:
        if detection.tag_id in [tag_id]:
            tag_matrix, _, _ = detector.detection_pose(detection, params)

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
        actual_tag_size = tag_size * (tag_modules_in_row / (tag_modules_in_row + 2))
        print("Tag pos: " + str(tag_pos))
        actual_tag_pos = tag_pos - actual_tag_size / 2  # position of the upper left corner of the tag
        print("Actual tag pos: " + str(actual_tag_pos))
        camera_pose[0:3] *= actual_tag_size

        camera_pose[0] += actual_tag_pos[0]
        camera_pose[1] += actual_tag_pos[1]
        camera_pose[2] += tag_pos[2]


        camera_matrix = pose2matrix(camera_pose)

        return camera_matrix
    else:
        return None





imagepath = '/home/bch_svt/cartpole/simulation/screenshots/screenshot.png'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
height, width = image.shape

camera_matrix = get_camera_pose(image, 255, 0.06 * scaler, scaler * np.array([-0.1, -0.1, 0]))

camera_pose = matrix2pose(camera_matrix)

print(np.array_str(camera_pose/one_millimeter, precision=3, suppress_small=True))

detector = apriltag.Detector()
detections, image = detector.detect(image, True)

cv2.imshow("pic", image)
cv2.waitKey(100000)

# camera params
fx, fy, sx, sy = get_camera_params(height, 45)


body_matrix = None
for detection in detections:
    if detection.tag_id in [300]:
        body_matrix, _, _ = detector.detection_pose(detection, [fx, fy, sx, sy])


body_matrix[0:3, 3] *= 0.06*scaler*0.8

print(body_matrix)
body_pose = camera_matrix @ body_matrix
print(np.array_str(matrix2pose(body_pose)/one_millimeter, precision=3, suppress_small=True))


