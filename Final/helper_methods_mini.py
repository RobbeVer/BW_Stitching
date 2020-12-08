import numpy as np
import os
import math
from scipy.optimize import shgo, minimize
import cv2
from PIL import Image

import pywt

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def calcRMSE(image1, image2):
    difference = image1-image2
    differenceSquared = difference**2
    meanOfDifferencesSquared = differenceSquared.mean()
    RMSE = np.sqrt(meanOfDifferencesSquared)
    return RMSE

def calcMI(image1, image2, bins = 20):
    hgram, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins)

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def correl_mismatch(x, y):
    x = np.array(x)
    y = np.array(y)
    x_mean0 = x.ravel() - x.mean()
    y_mean0 = y.ravel() - y.mean()
    corr_top = x_mean0.dot(y_mean0)
    corr_bottom = (np.sqrt(x_mean0.dot(x_mean0)) * np.sqrt(y_mean0.dot(y_mean0)))
    return -corr_top / corr_bottom

def paramstoaffine(params):
    # params = [trans_x, trans_y, scale_x, scale_y, sh_h, sh_v, angle]
    trans = np.identity(3)
    scale = np.identity(3)
    shear = np.identity(3)
    rot = np.zeros((3,3))

    trans[0,2] = params[0]
    trans[1,2] = params[1]

    scale[0,0] = params[2]
    scale[1,1] = params[3]

    shear[0,1] = params[4]
    shear[1,0] = params[5]

    angle = params[6]
    rot[0,0] = np.cos(angle)
    rot[0,1] = np.sin(angle)
    rot[1,0] = -np.sin(angle)
    rot[1,1] = np.cos(angle)

    M = shear.dot(rot).dot(scale).dot(trans)

    N = np.identity(3)

    if trans[0, 2] < 0:
        N[0, 2] = abs(trans[0, 2])
    if trans[1, 2] < 0:
        N[1, 2] = abs(trans[1, 2])

    return M,N

def warp_images(M,N):
    img_w, img_h = base_im.shape
    img_w = int(img_w + abs(M[1, 2]))
    img_h = int(img_h + abs(M[0, 2]))

    warped_base = cv2.warpAffine(base_im, N[0:2,:], (img_h, img_w))
    warped_next = cv2.warpAffine(next_im, M[0:2,:], (img_h, img_w))

    # warped_base = base_im.transform((img_w, img_h), Image.AFFINE, N_inv.flatten()[:6], resample=Image.NEAREST)
    # warped_next = next_im.transform((img_w, img_h), Image.AFFINE, M_inv.flatten()[:6], resample=Image.NEAREST)

    return warped_base, warped_next

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

def cost_function(params):
    """
    :param params: this is an array of the form [tf1, tf2, tf3, tf4, tf5, tf6]
    :return: returns the mismatch metric value
    """

    M, N = paramstoaffine(params)
    warped_base, warped_next = warp_images(M,N)

    return calcRMSE(warped_base, warped_next)
    """cA1, (cH1, cV1, cD1) = wavelettf_greyscale(warped_base, 'haar')
    cA2, (cH2, cV2, cD2) = wavelettf_greyscale(warped_next, 'haar')

    return calcRMSE(cA1, cA2)+calcRMSE(cH1, cH2)+calcRMSE(cV1, cV2)+calcRMSE(cD1, cD2)"""

def fusion(im1, im2, method):
    fused_im = np.zeros_like(im1)
    if (method == 'max'):
        for i in range(im1.shape[0]):
            for j in range(im1.shape[1]):
                fused_im[i,j] = max(im1[i,j], im2[i,j])
    elif (method == 'adding'):
        for i in range(fused_im.shape[0]):
            for j in range(fused_im.shape[1]):
                if im2[i][j] == 0:
                    fused_im[i][j] = im1[i][j]
                else:
                    fused_im[i][j] = im2[i][j]
    return fused_im

def Transforming(start_img_gray, next_img_gray):
    global base_im
    global next_im

    base_im = start_img_gray
    next_im = next_img_gray

    # trans_x, trans_y, scale_x, scale_y, sh_h, sh_v, angle
    bounds = [(-200, 200), (-200, 200), (0.95, 1.1), (0.95, 1.1), (-0.02, 0.02), (-0.02, 0.02), (-0.1, 0.1)]
    result = minimize(cost_function, [0,0,1,1,0,0,0], method='Powell', bounds=bounds, options={'maxiter': 50})
    best_params = result.x
    print(best_params)
    print(result.fun)
    best_M, best_N = paramstoaffine(best_params)
    warped_base, warped_next = warp_images(best_M, best_N)

    fused_im = fusion(warped_base, warped_next, 'adding')

    # Removing black borders
    fused_im = fused_im[~np.all(fused_im == 0, axis=1)]
    fused_im = fused_im[:, ~np.all(fused_im == 0, axis=0)]

    # cv2.imshow('test', warped_base)
    # cv2.imshow('test2', warped_next)
    # cv2.imshow('test3', fused_im)
    # cv2.waitKey()

    return fused_im