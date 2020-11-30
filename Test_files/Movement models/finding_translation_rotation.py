import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from skimage.transform import warp_polar, rotate, rescale
from scipy.ndimage import fourier_shift

path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'
image = cv2.imread(path_images + '\IMG_0781.JPG')
offset_image = cv2.imread(path_images + '\IMG_0782.JPG')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
offset_image_gray = cv2.cvtColor(offset_image, cv2.COLOR_BGR2GRAY)
image_down = None
offset_image_down = None
for i in range(0, 5):
    image_down = cv2.pyrDown(image_gray)
    offset_image_down = cv2.pyrDown(offset_image_gray)

shift, error, diffphase = phase_cross_correlation(image_down, offset_image_down)

radius = 705
image_polar = warp_polar(image_down, radius=radius, multichannel=False)
rotated_polar = warp_polar(offset_image_down, radius=radius, multichannel=False)
rotation, error, diffphase = phase_cross_correlation(image_polar, rotated_polar)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image_down, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image_down, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Show the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = np.fft.fft2(image_down) * np.fft.fft2(offset_image_down).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Cross-correlation")

print(f"Detected pixel offset (y, x): {shift}")
print(f"Detected a rotation of: {rotation[0]}")

plt.show()