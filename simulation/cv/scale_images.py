import cv2
import glob
from cv.camera import Camera
from constants import image_scale
import ntpath

src_folder = '/home/bch_svt/cartpole/simulation/cv/calibration_images_11cm'
dst_folder = '/home/bch_svt/cartpole/simulation/cv/calibration_images_11cm/scaled_05'

files = glob.glob(src_folder + '/*.bmp')
print(files)
ratio = 0.5
for f in files:
    im = cv2.imread(f)
    rescaled = Camera._scale_image(im, image_scale)
    filename = ntpath.basename(f)
    print(filename)
    cv2.imwrite(dst_folder + f'/{filename}', rescaled)
