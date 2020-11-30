from matplotlib import pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import os
import cv2

# Path to the images
path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'

image = cv2.imread(path_images + '\IMG_0781.JPG')

image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image_downscaled = rescale(image, 0.25, anti_aliasing=False)
# image_downscaled = downscale_local_mean(image, (15, 15, 1))
cv2.imwrite(path_images + '\\IMG_0781_downscaled.jpg', image_downscaled)
fig, axes = plt.subplots(nrows=1, ncols=2)

ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original image")

ax[1].imshow(image_downscaled, cmap='gray')
ax[1].set_title("Downscaled image (no aliasing)")

plt.tight_layout()
plt.show()