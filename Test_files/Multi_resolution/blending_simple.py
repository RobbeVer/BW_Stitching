import cv2
import os
import numpy as np

path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'
image = cv2.imread(path_images + '\IMG_0781.JPG')
offset_image = cv2.imread(path_images + '\IMG_0782.JPG')

# generate Gaussian pyramid for first image
nimage=image.copy()  
first=[image]
for i in range(4):
      nimage=cv2.pyrDown(nimage) # higher resolution to lower resolution for 4 levels 
      first.append(nimage) # pyramid are added in list
      
# generate Gaussian pyramid for second image (downsampling)
nimage1=offset_image.copy()
second=[offset_image]
for i1 in range(4):
        nimage1=cv2.pyrDown(nimage1)
        second.append(nimage1)

lpF = [first[3]] # one more list variable for Laplacian        
for i in range(3, 0, -1): # generate Laplacian Pyramid for first image
    size = (first[i - 1].shape[1],first[i - 1].shape[0]) # width and height of each pyramid
    first_expanded = cv2.pyrUp(first[i], dstsize=size) # lower to higher resolution
    laplacian = cv2.subtract(first[i - 1], first_expanded) # difference of original and expanded level
    lpF.append(laplacian)#added laplacian pyramid 

# laplacian pyramid for second image
lpS= [second[3]]
for i in range(3, 0, -1): # genrate Laplacian pyramid for second image
    size = (second[i - 1].shape[1],second[i - 1].shape[0])  # size of each level pyramid
    second_expanded = cv2.pyrUp(second[i], dstsize=size)
    laplacian = cv2.subtract(second[i - 1], second_expanded)
    lpS.append(laplacian)          

LS = [] # empty list for reconstructed image
n=0 # level counter
for la,lb in zip(lpF,lpS): 
    rows,cols,ch = la.shape # size and channels of image
    n=n+1
    ls = np.hstack((la[:, :int(cols/2)], lb[:, int(cols/2):])) # horizontally stack images 
    LS.append(ls) # added each level    
Image_reconstructed = LS[0] # reconstruction of final image

for i in range(1, 4):
    size = (LS[i].shape[1], LS[i].shape[0])
    Image_reconstructed = cv2.pyrUp(Image_reconstructed, dstsize=size)
    Image_reconstructed = cv2.add(LS[i],Image_reconstructed) # adding consecutive images
cv2.imwrite("blended_image.png",Image_reconstructed)   # showing new reconstructed image