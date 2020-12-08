#%% Imports + explanation
import cv2
import os
from queue import Queue

import helper_methods as hp

images = Queue(maxsize = 0)
length_progress = 0
i = 0

print('Loading images')
path_images = 'images'
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

print('Start stitching progress')
# for x in range(1):
#     image_A = cv2.pyrDown(image_A)
#     image_B = cv2.pyrDown(image_B)
hp.printProgressBar(0, length_progress, prefix='Progress', suffix='Complete', length=length_progress)
image_A = hp.Transforming(cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY))
i = 2
hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=length_progress)

cv2.imwrite('stitched_temp.jpg', image_A)

# %% Repeat steps until queue is empty
while (not (images.empty())):
    image_B = images.get()
    # for x in range(1):
    #     image_B = cv2.pyrDown(image_B)

    image_A = hp.Transforming(image_A, cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY))
    i += 1
    hp.printProgressBar(i, length_progress, prefix='Progress', suffix='Complete', length=length_progress)

    cv2.imwrite('stitched_temp.jpg', image_A)

# Save stitched image
cv2.imwrite('stitched_v1.jpg', image_A)
print('Finished')
