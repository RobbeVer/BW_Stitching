#%% Imports + explanation
import cv2
import os
from queue import Queue


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
# Path to the images
path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'
if os.path.isdir(path_images):
    entries = os.listdir(path_images)
    for entry in entries:
        images.put(cv2.imread(path_images + '\\' + entry))
        
image_A = images.get()
image_B = images.get()

#%% Get transform matrix from A to B


#%% Get translatation vector from transform matrix


#%% Move image B "translation vector"-amount
image_B_moved = 

#%% Stitch moved version of B with A => stitched_image


#%% Image A = original image B
image_A = image_B

#%% Repeat steps until queue is empty
while(not(images.empty()))
    image_B = images.get()
    
    # Get transformation matrix A to B
    
    # Get translation vector from transfromation matrix
    
    # Add translation vector to total translation vector
    
    # Move image B "total translation vector"-amount = image_B_moved
    image_B_moved = 

    # Stitch image B together with stitched_image
    
    # Image A = orginal image B
    image_A = image_B

#Plot stitched images