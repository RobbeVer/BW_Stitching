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
image = cv2.imread(path_images + '\IMG_0781.JPG')
offset_image = cv2.imread(path_images + '\IMG_0782.JPG')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
offset_image_gray = cv2.cvtColor(offset_image, cv2.COLOR_BGR2GRAY)
image_list = []
offset_image_list = []
for i in range(4):
    image_gray = cv2.pyrDown(image_gray)
    offset_image_gray = cv2.pyrDown(offset_image_gray)
    image_list.append(image_gray)
    offset_image_list.append(offset_image_gray)

# TODO: working with wavelets
coeffs = wavelettf_greyscale(image_list[0], 'haar')
coeffs2 = wavelettf_greyscale(offset_image_list[0], 'haar')

cA1, (cH1, cV1, cD1) = coeffs
cA2, (cH2, cV2, cD2) = coeffs2

radius = 705
image_polar = warp_polar(cV1, radius=radius, multichannel=False)
rotated_polar = warp_polar(cV2, radius=radius, multichannel=False)
rotation, error, diffphase = phase_cross_correlation(image_polar, rotated_polar)
print(rotation)

shift1, error, diffphase = phase_cross_correlation(cH1, cH2)
shift2, error1, diffphase1 = phase_cross_correlation(cV1, cV2)
print(f"Detected pixel offset in horizontal (y, x): {shift1}") # Take y-component from this one
print(f"Detected pixel offset in vertical (y, x): {shift2}") # Take x-component from this one

new_image =  np.zeros([image_list[0].shape[0]+2*int(abs(shift1[0])), image_list[0].shape[1]+2*int(abs(shift2[1]))],dtype=np.uint8)
new_offset_image =  np.zeros([image_list[0].shape[0]+2*int(abs(shift1[0])), image_list[0].shape[1]+2*int(abs(shift2[1]))],dtype=np.uint8)

new_image[2*int(abs(shift1[0]))-1:-1, 2*int(abs(shift2[1]))-1:-1] = image_list[0][:,:]
new_offset_image[0:image_list[0].shape[0], 0:image_list[0].shape[1]] = offset_image_list[0][:,:]

coeffs = wavelettf_greyscale(new_image, 'haar')
coeffs2 = wavelettf_greyscale(new_offset_image, 'haar')

fusedCoeffs = imageFusion(coeffs, coeffs2, 'max') # Better use max here for better results
fusedImage = pywt.waverec2(fusedCoeffs, 'haar')

fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
fusedImage = fusedImage.astype(np.uint8)

cv2.imshow("win",fusedImage)
cv2.waitKey()