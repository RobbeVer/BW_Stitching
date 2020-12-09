#%% Imports + explanation
import cv2
import os
import numpy as np
from queue import Queue
import helper_methods as hp
import copy

#%% Inital loading of images
images = Queue(maxsize = 0)
stitched_images = []
length_progress = 0
i = 0

print('\n\
    ██╗███╗   ███╗ █████╗  ██████╗ ███████╗    ███████╗████████╗██╗████████╗ ██████╗██╗  ██╗███████╗██████╗     ██╗   ██╗██████╗ \n\
    ██║████╗ ████║██╔══██╗██╔════╝ ██╔════╝    ██╔════╝╚══██╔══╝██║╚══██╔══╝██╔════╝██║  ██║██╔════╝██╔══██╗    ██║   ██║╚════██╗\n\
    ██║██╔████╔██║███████║██║  ███╗█████╗      ███████╗   ██║   ██║   ██║   ██║     ███████║█████╗  ██████╔╝    ██║   ██║ █████╔╝\n\
    ██║██║╚██╔╝██║██╔══██║██║   ██║██╔══╝      ╚════██║   ██║   ██║   ██║   ██║     ██╔══██║██╔══╝  ██╔══██╗    ╚██╗ ██╔╝██╔═══╝ \n\
    ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗    ███████║   ██║   ██║   ██║   ╚██████╗██║  ██║███████╗██║  ██║     ╚████╔╝ ███████╗\n\
    ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚══════╝   ╚═╝   ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝      ╚═══╝  ╚══════╝\n\
')

print('Loading images')
# Path to images
path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'
if os.path.isdir(path_images):
    entries = os.listdir(path_images)
    length_progress = len(entries)
    hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=100)
    for entry in entries:
        image = cv2.imread(path_images + '\\' + entry)
        image = cv2.pyrDown(image)
        image = cv2.pyrDown(image)
        images.put(image)
        i += 1
        hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=100)
print('Images loaded')

#%% Start the pre-stitching process
# Stich the images that we get from the queue
print('Start stitching progress')
i = 0
hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=100)

while(not(images.empty())):
    image_A = images.get()
    image_B = images.get()

    stitched_image = hp.Transforming(cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY))
    stitched_images.append(stitched_image)
    i += 2
    hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=100)

#%% Start the sub-stitching process
path_stitched_images = os.path.expanduser('~') + '\Pictures\Stitched' # used for previewing the stitched images
length_stitched_images = len(stitched_images)
counter = 0
sub = 1
while(length_stitched_images != 1):
    print('Start sub stitching progress', sub)
    for x in range(0, length_stitched_images, 2):
        hp.printProgressBar(x, length_stitched_images, prefix='Progress', suffix='Complete', length=100)
        if x+1 >= length_stitched_images-1:
            image_A = stitched_images[x]
            stitched_images[x//2] = image_A
            cv2.imwrite(path_stitched_images + '\stich_' + str(sub) + "_" + str(counter) + ".jpg", stitched_image)
            counter += 1
        else:
            image_A = stitched_images[x]
            image_B = stitched_images[x+1]

            stitched_image = hp.Transforming(image_A, image_B)
            stitched_images[x//2] = stitched_image
            cv2.imwrite(path_stitched_images + '\stich_' + str(sub) + "_" + str(counter) + ".jpg", stitched_image)
            counter += 1
    
    hp.printProgressBar(length_stitched_images, length_stitched_images, prefix='Progress', suffix='Complete', length=100)
    length_stitched_images = counter
    counter = 0
    sub += 1
    if length_stitched_images != 1:
        stitched_images = stitched_images[:length_stitched_images]

#%% Sharpen the image
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
stitched_images[0]  = cv2.filter2D(stitched_images[0], -1, kernel)

#%% Save the stitched image
cv2.imwrite('stitched_v2.jpg', stitched_images[0])
print('Finished')