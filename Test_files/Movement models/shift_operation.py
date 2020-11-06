# For rigid transformations

import cv2
import os
from image_registration import chi2_shift
from scipy.ndimage import shift
from matplotlib import pyplot as plt

# Path to the images
path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'

image = cv2.imread(path_images + '\IMG_0781.JPG')
offset_image = cv2.imread(path_images + '\IMG_0782.JPG')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
offset_image = cv2.cvtColor(offset_image, cv2.COLOR_RGB2GRAY)
noise = 0.1
xoff, yoff, exoff, eyoff = chi2_shift(image, offset_image, noise, return_error = True, upsample_factor = 'auto')

print('Pixels shifted by:', xoff, yoff)

corrected_image = shift(offset_image, shift=(xoff, yoff), mode='constant')

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected image')
plt.show()