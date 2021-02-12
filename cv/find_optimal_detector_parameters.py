from cv.camera import Camera
import dt_apriltags
import cv2
import numpy as np

im = cv2.imread('../debug_images/image_5.jpg', cv2.IMREAD_GRAYSCALE)


sigma = 1.0
decimate = 1.0
sharpening = 1.0
threads = 8



for sigma in np.arange(0.0, 2.0, 0.05):
    for sharpening in (0.0, 1.0, 0.05):
        detector = dt_apriltags.Detector(nthreads=threads,
                                             quad_decimate=decimate,
                                             quad_sigma=sigma,
                                             decode_sharpening=sharpening)

        detections = detector.detect(im)

        if len(detections) > 48:
            print(f"Tada!")

        for detection in detections:
            if detection.tag_id == 102 or detection.tag_id == 103:
                print(f"Bingo!")

        # print(f"Param = {sharpening}, Len: {len(detections)}")


