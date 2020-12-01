import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from skimage.transform import warp_polar, rotate, rescale
from scipy.ndimage import fourier_shift

def calcMSE(image1, image2): #Calc MSE of greyscale --> Higher value = less similar
    err_matrix = (image1.astype("float") - image2.astype("float"))**2
    err = np.sum(err_matrix)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'
image = cv2.imread(path_images + '\IMG_0781.JPG')
offset_image = cv2.imread(path_images + '\IMG_0782.JPG')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
offset_image_gray = cv2.cvtColor(offset_image, cv2.COLOR_BGR2GRAY)
image_list = []
offsetImage_list = []

for i in range(0, 5):
    image_gray = cv2.pyrDown(image_gray)
    offset_image_gray = cv2.pyrDown(offset_image_gray)
    image_list.append(cv2.pyrDown(image_gray))
    offsetImage_list.append(cv2.pyrDown(offset_image_gray))

radius = 705

for i in range(len(image_list)):  
    shift, error, diffphase = phase_cross_correlation(image_list[i], offsetImage_list[i])

    image_polar = warp_polar(image_list[i], radius=radius, multichannel=False)
    rotated_polar = warp_polar(offsetImage_list[i], radius=radius, multichannel=False)
    rotation, error, diffphase = phase_cross_correlation(image_polar, rotated_polar)    

    print(calcMSE(image_list[i], offsetImage_list[i]))
    print(f"Detected pixel offset (y, x): {shift}")
    print(f"Detected a rotation of: {rotation[0]}")

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image_list[0], cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offsetImage_list[0], cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Show the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = np.fft.fft2(image_list[0]) * np.fft.fft2(offsetImage_list[0]).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Cross-correlation")

plt.show()