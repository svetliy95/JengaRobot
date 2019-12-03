import cv2
import numpy as np
import apriltag
import time

imagepath = '/home/bch_svt/cartpole/cv/pictures/Screenshot from 2019-12-02 17-09-51.png'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)

# cv2.imshow("pic", image)
# cv2.waitKey(100000)

detector = apriltag.Detector()


start = time.time()
detections, image = detector.detect(image, True)
stop = time.time()
print(stop - start)

# cv2.imshow("pic", image)
# cv2.waitKey(100000)


print([detections[0]])