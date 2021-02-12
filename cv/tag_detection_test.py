from cv.camera import Camera
from constants import *
import dt_apriltags
import math

cam1 = Camera(cam1_serial, cam1_mtx, cam1_dist)
cam1.start_grabbing()
im = cam1.get_raw_image()

detector = dt_apriltags.Detector(nthreads=8, quad_decimate=2)

detections = detector.detect(im, True, cam1.get_params(), 1)

print(f"Detections: \n")
for d in detections:
    if d.tag_id == 255:
        detection = d

quat = Quaternion(matrix=detection.pose_R)
# quat = quat.inverse
euler = list(map(math.degrees, quat.yaw_pitch_roll[::-1]))
print(euler)



