import cv2
import numpy as np
import apriltag
import time
import math
from transformations import matrix2pose

imagepath = '/home/bch_svt/cartpole/cv/pictures/Screenshot from 2019-12-02 17-09-51.png'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
height, width = image.shape

detector = apriltag.Detector()
detections, image = detector.detect(image, True)

# cv2.imshow("pic", image)
# cv2.waitKey(100000)

f = 0.5 * height / math.tan(45 * math.pi / 360)
fx = f
fy = f
sx = width / 2
sy = height / 2

pose = None
for detection in detections:
    if detection.tag_id == 255:
        mtx, _, _ = detector.detection_pose(detection, [fx, fy, sx, sy])
        pose = matrix2pose(mtx)

print(pose)