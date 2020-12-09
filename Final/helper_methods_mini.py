import numpy as np
from scipy.optimize import minimize
import cv2
import pywt

"""
This file contains all the helper functions needed for the general algorithm in main.py. Each function contains
an explanation about the functionality and parameters.
"""

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Creates terminal progress bar when called in a loop and prints it.
    :param iteration   - Required  : current iteration (Int)
    :param total       - Required  : total iterations (Int)
    :param prefix      - Optional  : prefix string (Str)
    :param suffix      - Optional  : suffix string (Str)
    :param decimals    - Optional  : positive number of decimals in percent complete (Int)
    :param length      - Optional  : character length of bar (Int)
    :param fill        - Optional  : bar fill character (Str)
    :param printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)

    :return
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def calcRMSE(image1, image2):
    """
    Calculates the Root Mean Squared Error (RMSE) given two images of equal size.
    :param image1:      First image of the two images (ndarray)
    :param image2:      Second image of the two images (ndarray)

    :return: RMSE       The RMSE value of the two images (float64)
    """
    # %% Checking if image dimensions match
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same size")

    # %% Calculating RMSE and returning it
    difference = image1-image2
    differenceSquared = difference**2
    meanOfDifferencesSquared = differenceSquared.mean()
    RMSE = np.sqrt(meanOfDifferencesSquared)
    return RMSE

def paramstoaffine(params):
    """
    Transforms the parameters into an affine transformation matrix.
    :param params:      An array containing the needed parameters. The array must be of the form
                         [trans_x, trans_y, scale_x, scale_y, sh_h, sh_v, angle] (ndarray)

    :return: M          A 3x3 affine transformation matrix based on params (ndarray)
    :return: N          A 3x3 affine transformation matrix containing a translation (ndarray)
    """
    # %% Checking if params has right length
    if len(params) != 7:
        raise ValueError("Parameter array should contain 7 values")

    # %% Initializing the transformation matrices of the different transformations.
    trans = np.identity(3)
    scale = np.identity(3)
    shear = np.identity(3)
    rot = np.zeros((3, 3))

    # %% Inserting the parameters in the right place in the right matrices.
    trans[0, 2] = params[0]
    trans[1, 2] = params[1]
    scale[0, 0] = params[2]
    scale[1, 1] = params[3]
    shear[0, 1] = params[4]
    shear[1, 0] = params[5]
    angle = params[6]
    rot[0, 0] = np.cos(angle)
    rot[0, 1] = np.sin(angle)
    rot[1, 0] = -np.sin(angle)
    rot[1, 1] = np.cos(angle)

    # %% Multiplying the different rotation matrices to become total transformation matrix M
    M = shear.dot(rot).dot(scale).dot(trans)

    # %% Constructing a translation matrix N needed to possibly translate the stitched image
    N = np.identity(3)
    if trans[0, 2] < 0:
        N[0, 2] = abs(trans[0, 2])
    if trans[1, 2] < 0:
        N[1, 2] = abs(trans[1, 2])

    return M, N

def warp_images(M,N):
    """
    Transforms the images according to the given transformation matrices.
    :param M:             Affine transformation matrix that transforms the next image (ndarray)
    :param N:             Affine transformation matrix that translates the base image (ndarray)

    :return: warped_base: Transformed base image (ndarray)
    :return: warped_base: Transformed next image (ndarray)
    """
    # %% Determining the shape the warped images should have
    img_w, img_h = base_im.shape
    img_w = int(img_w + abs(M[1, 2]))
    img_h = int(img_h + abs(M[0, 2]))

    # %% Transforming the images and returning them
    warped_base = cv2.warpAffine(base_im, N[0:2,:], (img_h, img_w))
    warped_next = cv2.warpAffine(next_im, M[0:2,:], (img_h, img_w))
    return warped_base, warped_next

def cost_function(params):
    """
    Calculates the cost metric after transforming according to the given parameters.
    :param params           An array containing the parameters to optimize. The array must be of the form
                            [trans_x, trans_y, scale_x, scale_y, sh_h, sh_v, angle] (ndarray)

    :return: RMSE           The RMSE of the transformed images
    """
    # %% Transforming the images according to the parameters
    M, N = paramstoaffine(params)
    warped_base, warped_next = warp_images(M,N)

    # %% Calculating the RMSE and returning it
    RMSE = calcRMSE(warped_base, warped_next)
    return RMSE

def fusion(im1, im2, method):
    """
    Fuses the two images together following a certain method
    :param im1:             The first image of the two to be fused (ndarray)
    :param im2:             The second image of the two to be fused (ndarray)
    :param method:          Fusing method (String)
                                'max' -> selects the maximum value of the pixel
                                'adding' -> selects pixel value of image 1 if value of image 2 is zero

    :return: fused_im       Fused image of the rwo source images (ndarray)
    """
    # %% Checking if appropriate method is given
    if method != 'max' and method != 'adding':
        raise ValueError("No appropriate method is given")

    # %% Fusing the two images together using the given method
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
    """
    Finds the optimal parameter, transforms the images accordingly and fuses them together.
    :param start_img_gray:  First of the two images to register and fuse together
    :param next_img_gray:   Second of the two images to register and fuse together

    :return: fused_im       Fused image of the two source images
    """
    # %% Creating global variables to be used in other functions containing the base image and the next image
    global base_im
    global next_im
    base_im = start_img_gray
    next_im = next_img_gray

    # %% Optimizing the parameters to find the optimal transformation
    bounds = [(-200, 200), (-200, 200), (0.95, 1.1), (0.95, 1.1), (-0.02, 0.02), (-0.02, 0.02), (-0.1, 0.1)]
    result = minimize(cost_function, [0,0,1,1,0,0,0], method='Powell', bounds=bounds, options={'maxiter': 50})
    best_params = result.x

    # %% Transforming the images according to the optimal parameters.
    best_M, best_N = paramstoaffine(best_params)
    warped_base, warped_next = warp_images(best_M, best_N)

    # %% Fusing the images together
    fused_im = fusion(warped_base, warped_next, 'max')

    # %% Removing black borders
    fused_im = fused_im[~np.all(fused_im == 0, axis=1)]
    fused_im = fused_im[:, ~np.all(fused_im == 0, axis=0)]

    return fused_im