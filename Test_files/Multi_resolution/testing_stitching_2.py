import numpy as np
import os
import math
from matplotlib import pyplot as plt

import cv2

import pywt

def findDimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0,0]
    base_p2[:2] = [x,0]
    base_p3[:2] = [0,y]
    base_p4[:2] = [x,y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:
        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

        if ( max_x == None or normal_pt[0,0] > max_x ):
            max_x = normal_pt[0,0]

        if ( max_y == None or normal_pt[1,0] > max_y ):
            max_y = normal_pt[1,0]

        if ( min_x == None or normal_pt[0,0] < min_x ):
            min_x = normal_pt[0,0]

        if ( min_y == None or normal_pt[1,0] < min_y ):
            min_y = normal_pt[1,0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)

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

def fuseCoeff2(coeffs1, coeffs2, method):
    cA = []
    cH = []
    cV = []
    cD = []
    indices = []
    if method == 'adding':
        cA = coeffs2[0]
        cH = coeffs2[1][0]
        cV = coeffs2[1][1]
        cD = coeffs2[1][2]
        print(len(coeffs1[0][0]))
        for i in range(coeffs1[0].shape[0]):
            indices = np.where(coeffs2[0][i] >= 0)
            cA[i][indices] = coeffs1[0][i][indices]
            cH[i][indices] = coeffs1[1][0][i][indices]
            cV[i][indices] = coeffs1[1][1][i][indices]
            cD[i][indices] = coeffs1[1][2][i][indices]
    return [cA, (cH, cV, cD)]

def imageFusion(coeffs1, coeffs2, method):
    fusedCoeffs = []
    for i in range(len(coeffs1)-1):

        # The first values in each decomposition is the approximation values of the top level
        if(i == 0):
            fusedCoeffs.append(fuseCoeff(coeffs1[0],coeffs2[0], method))
        else:
            # For the rest of the levels we have tupels with 3 coeeficents
            c1 = fuseCoeff(coeffs1[i][0], coeffs2[i][0], method)
            c2 = fuseCoeff(coeffs1[i][1], coeffs2[i][1], method)
            c3 = fuseCoeff(coeffs1[i][2], coeffs2[i][2], method)

            fusedCoeffs.append((c1,c2,c3))
    return fusedCoeffs

def wavelettf_greyscale(img, wavelet):
    """
    Function that first converts the image to greyscale and then calculates the wavelet coeffs
    :param img: original image, size is (M N 3)
    :param wavelet: kind of wavelet used, string
    :return: the different wavelet coefficients
    """

    # conversion to right data type + to greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.float32(img)
    img /= 255

    # extracting the coeffs
    coeffs = pywt.dwt2(img, wavelet, axes=(0, 1))
    cA, (cH, cV, cD) = coeffs
    return coeffs

def plot_coeffs(coeffs, title):
    cA, (cH, cV, cD) = coeffs

    fig = plt.figure(figsize=(30, 30))

    plt.subplot(2, 2, 1)
    plt.imshow(cA, cmap=plt.cm.gray)
    plt.title('cA', fontsize=30)
    plt.subplot(2, 2, 2)
    plt.imshow(cH, cmap=plt.cm.gray)
    plt.title('cH', fontsize=30)
    plt.subplot(2, 2, 3)
    plt.imshow(cV, cmap=plt.cm.gray)
    plt.title('cV', fontsize=30)
    plt.subplot(2, 2, 4)
    plt.imshow(cD, cmap=plt.cm.gray)
    plt.title('cD', fontsize=30)

    fig.suptitle(title, fontsize=60)
    plt.show()

path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'

# Load starting image
start_img = cv2.imread(path_images + '\IMG_0783.JPG')
start_img_gray = cv2.cvtColor(start_img, cv2.COLOR_BGR2GRAY)

# Downsample the image
start_img_gray = cv2.pyrDown(start_img_gray)

coeffs = wavelettf_greyscale(start_img_gray, 'haar')
cA, (cH, cV, cD) = coeffs
scale_val_H = 255 / np.max(cH)
scale_val_V = 255 / np.max(cV)
cH = cH * scale_val_H
cV = cV * scale_val_V
# Find features
sift = cv2.SIFT_create()

# print(np.max(cH))
kp1, des1 = sift.detectAndCompute(np.uint8((cH + cV)), None)

# Parameters for nearest-neighbor matching
FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params)

# Get the next image
next_img = cv2.imread(path_images + '\IMG_0784.JPG')
next_img_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

# Downsample the image
next_img_gray = cv2.pyrDown(next_img_gray)

coeffs2 = wavelettf_greyscale(next_img_gray, 'haar')
cA2, (cH2, cV2, cD2) = coeffs2

# Find points in the next frame

scale_val_H = 255 / np.max(cH2)
scale_val_V = 255 / np.max(cV2)
cH2 = cH2 * scale_val_H
cV2 = cV2 * scale_val_V
kp2, des2 = sift.detectAndCompute(np.uint8((cH2 + cV2)), None)

matches = matcher.knnMatch(des1, des2, k=2)
good = []
for m,n in matches: # m is from the first image, n is from the next image
    if m.distance < 0.8*n.distance:
        good.append(m)

# If we want to see the matches we can uncomment this section
# color = (250, 128, 114)
# img_match = cv2.drawMatches(start_img_gray, kp1, next_img_gray, kp2, good, outImg=None, matchColor=color, singlePointColor=color)
# cv2.imshow('Test', img_match)
# cv2.waitKey()

if len(good) > 10:
    src_pts = np.array([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good])

    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # H = H / H[2,2]
    H_inv = np.linalg.inv(H)

    (min_x, min_y, max_x, max_y) = findDimensions(cv2.pyrDown(next_img_gray), H_inv)

    max_x = max(max_x, cv2.pyrDown(start_img_gray).shape[1])
    max_y = max(max_y, cv2.pyrDown(start_img_gray).shape[0])

    move_h = np.matrix(np.identity(3), np.float32)

    if ( min_x < 0 ):
        move_h[0,2] += -min_x
        max_x += -min_x

    if ( min_y < 0 ):
        move_h[1,2] += -min_y
        max_y += -min_y

    # print ("Homography: \n", H)
    # print ("Inverse Homography: \n", H_inv)
    # print ("Min Points: ", (min_x, min_y))

    mod_inv_h = move_h * H_inv
    
    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))

base_img_warp = cv2.warpPerspective(cv2.pyrDown(start_img_gray), move_h, (img_w, img_h))
next_img_warp = cv2.warpPerspective(cv2.pyrDown(next_img_gray), mod_inv_h, (img_w, img_h))

# cv2.imshow("Base",base_img_warp)
# cv2.imshow("Next",next_img_warp)

# cv2.waitKey()

# First method for fusing
coeffs = wavelettf_greyscale(base_img_warp, 'haar')
coeffs2 = wavelettf_greyscale(next_img_warp, 'haar')

cA1, (cH1, cV1, cD1) = coeffs
cA2, (cH2, cV2, cD2) = coeffs2

fusedCoeffs = imageFusion(coeffs, coeffs2, 'max') # Better use max here for better results

# plot_coeffs(fusedCoeffs, 'Fused')

fusedImage = pywt.waverec2(fusedCoeffs, 'haar')

fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
fusedImage = fusedImage.astype(np.uint8)

cv2.imshow("Fused",fusedImage)

cv2.waitKey()