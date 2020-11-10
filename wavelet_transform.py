import numpy as np
import pywt
import pywt.data
import cv2
from matplotlib import pyplot as plt

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

def wavelettf_color(img, wavelet):
    """
    Function that treats each color channel on its own
    :param img: original image, size is (M N 3)
    :param wavelet: kind of wavelet used, string
    :return: list with the different wavelet coefficients for each color channel
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #separation of color channels
    coeffs_list = []
    for i in range(3):
        curr_img = img[:, :, i]
        curr_img = np.float32(curr_img)
        curr_img /= 255

        coeffs = pywt.dwt2(curr_img, wavelet, axes=(0, 1))
        cA, (cH, cV, cD) = coeffs
        coeffs_list.append(coeffs)

    return coeffs_list


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
