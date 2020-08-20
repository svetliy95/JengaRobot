import dt_apriltags
from constants import *
import cv2
from cv.camera import Camera


if __name__ == "__main__":
    decimate = 1
    sigma = 1
    detector = dt_apriltags.Detector(quad_decimate=quad_decimate, quad_sigma=quad_sigma, nthreads=8, decode_sharpening=1)

    im = cv2.imread('../debug_images/image_5.jpg', cv2.IMREAD_GRAYSCALE)

    cam = Camera(cam2_serial, cam2_mtx, cam2_dist)

    detections = detector.detect(im, True, cam.get_params(), 1)

    is_there = False
    for d in detections:
        if d.tag_id == 103:
            is_there = True
            break

    # print(f"Detections: {detections}")
    print(f"Detection length: {len(detections)}")
    print(f"Is #13 there: {is_there}")