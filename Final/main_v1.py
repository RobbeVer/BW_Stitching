#%% Imports + explanation
import cv2
import os
from queue import Queue
import numpy as np
import helper_methods as hp

"""
Algorithm outline
    1. Read all images into a queue and downsample them.
    2. Take first two images out of the queue (image_A, image_B).
    3. Stitch the first two images together using hp.Transforming(image_A, image_B). 
        -> More details in helper_methods.py.
    4. stitched_image = stitched image returned from hp.Transforming()
    5. Repeat the queue until it is empty:
        5.1. Load next picture from queue and load into image_B
        5.2. Call hp.Transforming(stitched_image, image_B)
        5.3. stitched_image = stitched image returned from hp.Transforming()
    6. Save stitched_image into stitched_v1.jpg
"""

#%% Inital loading of images
images = Queue(maxsize = 0)
length_progress = 0
i = 0
# Path to the images

print('\n\
    ██╗███╗   ███╗ █████╗  ██████╗ ███████╗    ███████╗████████╗██╗████████╗ ██████╗██╗  ██╗███████╗██████╗     ██╗   ██╗ ██╗\n\
    ██║████╗ ████║██╔══██╗██╔════╝ ██╔════╝    ██╔════╝╚══██╔══╝██║╚══██╔══╝██╔════╝██║  ██║██╔════╝██╔══██╗    ██║   ██║███║\n\
    ██║██╔████╔██║███████║██║  ███╗█████╗      ███████╗   ██║   ██║   ██║   ██║     ███████║█████╗  ██████╔╝    ██║   ██║╚██║\n\
    ██║██║╚██╔╝██║██╔══██║██║   ██║██╔══╝      ╚════██║   ██║   ██║   ██║   ██║     ██╔══██║██╔══╝  ██╔══██╗    ╚██╗ ██╔╝ ██║\n\
    ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗    ███████║   ██║   ██║   ██║   ╚██████╗██║  ██║███████╗██║  ██║     ╚████╔╝  ██║\n\
    ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚══════╝   ╚═╝   ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝      ╚═══╝   ╚═╝\n\
    ')

print('Loading images')
path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'
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
image_A = images.get()
image_B = images.get()

#%% Start the stitching process
# Stich the images that we get from the queue
print('Start stitching progress')
hp.printProgressBar(0, length_progress, prefix='Progress', suffix='Complete', length=length_progress)
image_A = hp.Transforming(cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY))
i = 2
hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=length_progress)

#%% Repeat steps until queue is empty
while(not(images.empty())):
    image_B = images.get()

    image_A = hp.Transforming(image_A, cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY))
    i += 1
    hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=length_progress)
    

# Sharpen the image
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
image_A = cv2.filter2D(image_A, -1, kernel)

# Save stitched image
cv2.imwrite('stitched_v1.jpg', image_A)
print('Finished')