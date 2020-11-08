from matplotlib import pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from image_registration import cross_correlation_shifts
from scipy.ndimage import shift
import os
import cv2

# Path to the images
path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'

image = cv2.imread(path_images + '\IMG_0781.JPG')
lr = cv2.pyrDown(image)
cv2.imshow('Image', image)
cv2.imshow('lr', lr)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

cv2.waitKey()