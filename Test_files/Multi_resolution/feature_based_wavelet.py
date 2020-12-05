import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
from wavelet_transform import wavelettf_greyscale, wavelettf_color, plot_coeffs

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from skimage.transform import warp_polar, rotate, rescale
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift

import pywt

def calcMSE(image1, image2): # Calc MSE of greyscale --> Higher value = less similar
    err_matrix = (image1.astype("float") - image2.astype("float"))**2
    err = np.sum(err_matrix)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def fuseCoeff(coeffs1, coeffs2, method):
    if (method == 'mean'):
        coeff = (coeffs1 + coeffs2) / 2
    elif (method == 'min'):
        coeff = np.minimum(coeffs1, coeffs2)
    elif (method == 'max'):
        coeff = np.maximum(coeffs1, coeffs2)
    else:
        coeff = []
    return coeff

def imageFusion(coeffs1, coeffs2, method):
    fusedCoeffs = []
    for i in range(len(coeffs1)-1):

        # The first values in each decomposition is the apprximation values of the top level
        if(i == 0):
            fusedCoeffs.append(fuseCoeff(coeffs1[0],coeffs2[0], method))

        else:
            # For the rest of the levels we have tupels with 3 coeeficents
            c1 = fuseCoeff(coeffs1[i][0], coeffs2[i][0], method)
            c2 = fuseCoeff(coeffs1[i][1], coeffs2[i][1], method)
            c3 = fuseCoeff(coeffs1[i][2], coeffs2[i][2], method)

            fusedCoeffs.append((c1,c2,c3))
    return fusedCoeffs

path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'
image = cv2.imread(path_images + '\IMG_0783.JPG')
offset_image = cv2.imread(path_images + '\IMG_0784.JPG')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
offset_image_gray = cv2.cvtColor(offset_image, cv2.COLOR_BGR2GRAY)
image_list = []
offset_image_list = []
for i in range(4): # gaussian pyramid
    image_gray = cv2.pyrDown(image_gray)
    offset_image_gray = cv2.pyrDown(offset_image_gray)
    image_list.append(image_gray)
    offset_image_list.append(offset_image_gray)

# Find features
sift = cv2.SIFT_create()
# find the key points and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(image_list[1],None)
kp2, des2 = sift.detectAndCompute(offset_image_list[1],None)

match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.2*n.distance:
        good.append(m)

M = None
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h,w = offset_image_list[1].shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

dst = cv2.warpPerspective(offset_image_list[1],M,(image_list[1].shape[1] + offset_image_list[1].shape[1], image_list[1].shape[0])) # Transformation used here

dst_clitted = copy.deepcopy(dst)

cv2.imshow("dst", dst_clitted)

dst_clitted[0:image_list[1].shape[0],0:image_list[1].shape[1]] = image_list[1] # Clitting here

# Determine wavelet coefficients
coeffs = wavelettf_greyscale(image_list[0], 'haar')
coeffs2 = wavelettf_greyscale(offset_image_list[0], 'haar')

cA1, (cH1, cV1, cD1) = coeffs
cA2, (cH2, cV2, cD2) = coeffs2

# cv2.imshow("original_image_stitched.jpg", dst)
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

cv2.imshow("dst_clitted", trim(dst_clitted))
cv2.waitKey()
print(trim(dst_clitted).shape)
# cv2.imsave("original_image_stitched_crop.jpg", trim(dst))

# new_image =  np.zeros([image_list[0].shape[0]+2*int(abs(shift1[0])), image_list[0].shape[1]+2*int(abs(shift2[1]))],dtype=np.uint8)
# new_offset_image =  np.zeros([image_list[0].shape[0]+2*int(abs(shift1[0])), image_list[0].shape[1]+2*int(abs(shift2[1]))],dtype=np.uint8)

# # Look in which way they are moved from each other
# if shift1[0] < 0 and shift2[1] < 0:
#     new_image[2*int(abs(shift1[0]))-1:-1, 2*int(abs(shift2[1]))-1:-1] = image_list[0][:,:]
#     new_offset_image[0:image_list[0].shape[0], 0:image_list[0].shape[1]] = offset_image_list[0][:,:]
# elif shift1[0] > 0 and shift2[1] < 0:
#     new_image[2*int(abs(shift1[0]))-1:-1, 2*int(abs(shift2[1]))-1:-1] = image_list[0][:,:]
#     new_offset_image[0:image_list[0].shape[0], 0:image_list[0].shape[1]] = offset_image_list[0][:,:]
# elif shift1[0] < 0 and shift2[1] > 0:
#     new_image[2*int(abs(shift1[0]))-1:-1, 2*int(abs(shift2[1]))-1:-1] = image_list[0][:,:]
#     new_offset_image[0:image_list[0].shape[0], 0:image_list[0].shape[1]] = offset_image_list[0][:,:]    
# else:
#     new_image[0:image_list[0].shape[0], 0:image_list[0].shape[1]] = image_list[0][:,:]
#     new_offset_image[2*int(abs(shift1[0]))-1:-1, 2*int(abs(shift2[1]))-1:-1] = offset_image_list[0][:,:]

# coeffs = wavelettf_greyscale(new_image, 'haar')
# coeffs2 = wavelettf_greyscale(new_offset_image, 'haar')

# cA1, (cH1, cV1, cD1) = coeffs
# cA2, (cH2, cV2, cD2) = coeffs2

# fig = plt.figure(figsize=(30, 30))
# plt.subplot(2, 4, 1)
# plt.imshow(cA1, cmap=plt.cm.gray)
# plt.title('cA1', fontsize=10)

# plt.subplot(2, 4, 2)
# plt.imshow(cH1, cmap=plt.cm.gray)
# plt.title('cH1', fontsize=10)

# plt.subplot(2, 4, 5)
# plt.imshow(cV1, cmap=plt.cm.gray)
# plt.title('cV1', fontsize=10)

# plt.subplot(2, 4, 6)
# plt.imshow(cD1, cmap=plt.cm.gray)
# plt.title('cD1', fontsize=10)

# plt.subplot(2, 4, 3)
# plt.imshow(cA2, cmap=plt.cm.gray)
# plt.title('cA2', fontsize=10)

# plt.subplot(2, 4, 4)
# plt.imshow(cH2, cmap=plt.cm.gray)
# plt.title('cH2', fontsize=10)

# plt.subplot(2, 4, 7)
# plt.imshow(cV2, cmap=plt.cm.gray)
# plt.title('cV2', fontsize=10)

# plt.subplot(2, 4, 8)
# plt.imshow(cD2, cmap=plt.cm.gray)
# plt.title('cD2', fontsize=10)

# fig.suptitle('Testing wavelets', fontsize=20)
# plt.show()

# cA2 = rotate(cA2, -(rotation[0]))
# cH2 = rotate(cH2, -(rotation[0]))
# cV2 = rotate(cV2, -(rotation[0]))
# cD2 = rotate(cD2, -(rotation[0]))
# coeffs2_rotated = (cA2, (cH2, cV2, cD2))
# fusedCoeffs = imageFusion(coeffs, coeffs2_rotated, 'max') # Better use max here for better results
# fusedImage = pywt.waverec2(fusedCoeffs, 'haar')

# fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
# fusedImage = fusedImage.astype(np.uint8)

# cv2.imshow("win",fusedImage)
# cv2.waitKey()