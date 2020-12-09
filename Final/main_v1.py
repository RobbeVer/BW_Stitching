#%% Imports + explanation
import cv2
import os
from queue import Queue
import numpy as np
import helper_methods as hp

# 1. Lees foto A en foto B in
# 2. Functie X op beide foto's --> Krijg transformatie matrix
# 3. Haal translatievector uit transformatie matrix
# 4. Beweeg foto B met bewegingsvector
# 5. Stitch foto A en bewogen foto B = stitched_image
# 6. foto A = orginele foto B
# 7. Do tot alle foto's ingeladen zijn:
#       Laad volgende foto B in
#       Functie X op foto A en foto B --> Krijg transformatie matrix
#       Haal translatievector uit transformatie matrix
#       bewegingsvector_totaal += bewegingsvector
#       Beweeg foto B met bewegingsvector_totaal
#       Stitch stitched_foto met bewogen foto B.
#       foto A = orginele foto B

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

#%% Get transform matrix from A to B


#%% Get translatation vector from transform matrix


#%% Move image B "translation vector"-amount
# image_B_moved = 

#%% Stitch moved version of B with A => stitched_image


#%% Image A = original image B
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