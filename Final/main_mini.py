#%% Imports + explanation
import cv2
import os
from queue import Queue
import helper_methods as hp

"""
Algorithm outline
    1. Read all images into a queue and downsample them.
    2. Take first two images out of the queue (image_A, image_B).
    3. Stitch the first two images together using hp.Transforming(image_A, image_B). 
        -> More details in helper_methods_minimization.py.
    4. stitched_image = stitched image returned from hp.Transforming()
    5. Repeat the queue is empty:
        5.1. Load next picture from queue into image_B
        5.2. Call hp.Transforming(stitched_image, image_B)
        5.3. stitched_image = stitched image returned from hp.Transforming()
    6. Save stitched_image into stitched_minimization.jpg
"""

# %% Loading images into a Queue and downsample them
print('Loading images')
images = Queue(maxsize = 0)
length_progress = 0
i = 0

path_images = 'images' #os.path.expanduser('~') + '\Pictures\Stitching_images'
if os.path.isdir(path_images):
    entries = os.listdir(path_images)
    length_progress = len(entries)
    hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=length_progress)
    for entry in entries:
        image = cv2.imread(path_images + '\\' + entry)
        image = cv2.pyrDown(image)
        image = cv2.pyrDown(image)
        images.put(image)
        i += 1
        hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=length_progress)
print('Images loaded')


# %% Start stitching process with stitching of initial two images => stitched image into
print('Start stitching process')
image_A = images.get()
image_B = images.get()
hp.printProgressBar(0, length_progress, prefix='Progress', suffix='Complete', length=length_progress)
stitched_image = hp.Transforming(cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY))
i = 2
hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=length_progress)


# %% Stitch next image to the existing image untill the queue is empty
while (not (images.empty())):
    image_B = images.get()

    stitched_image = hp.Transforming(stitched_image, cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY))
    i += 1
    hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=length_progress)


# %% Save stitched image
cv2.imwrite('stitched_minimization.jpg', stitched_image)
print('Finished')
