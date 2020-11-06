# For non rigid transformations
# Takes a long time for computing it

import cv2
import os
import numpy as np
from image_registration import cross_correlation_shifts
from scipy.ndimage import shift
from matplotlib import pyplot as plt
from skimage import registration

# Path to the images
path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'

image = cv2.imread(path_images + '\IMG_0781.JPG')
offset_image = cv2.imread(path_images + '\IMG_0782.JPG')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
offset_image = cv2.cvtColor(offset_image, cv2.COLOR_RGB2GRAY)

flow = registration.optical_flow_tvl1(image, offset_image)

# Display dense optical flow
flow_x = flow[1, :, :]
flow_y = flow[0, :, :]

# Let us find the mean of all pixels in x and y and shift image by that
# Ideally, you need to move each pixel by the amount from flow
xoff = np.mean(flow_x)
yoff = np.mean(flow_y)

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