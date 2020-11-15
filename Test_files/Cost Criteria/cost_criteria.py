from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math

def calcRMSE(image1, image2):
    difference = image1 - image2
    differenceSquared = difference**2
    meanOfDifferencesSquared = differenceSquared.mean()
    RMSE = np.sqrt(meanOfDifferencesSquared)
    return RMSE


def calcMSE(image1, image2): #Calc MSE of greyscale --> Higher value = less similar
    err_matrix = (image1.astype("float") - image2.astype("float"))**2
    err = np.sum(err_matrix)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def calcPSNR(image1, image2):
    MSE = calcMSE(image1, image2)
    return 10*math.log10((255**2)/(MSE))
    

def calcSSIM(image1, image2): #Calc SSIM of greyscale --> Higher value = more similar
    return ssim(image1,image2)

#Load images from files:
path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Pictures\\")
images = []
for i in range(781,784): #Just 3 images for testing
    print(path+"IMG_0"+str(i)+".JPG")
    image = cv2.imread(path+"IMG_0"+str(i)+".JPG")
    images.append(image)
images = np.array(images)
    
values = []
for i in range(images.shape[0]):
    if i < 3: #Just compare for first 3 images
        original = images[i]
        for j,other in enumerate(images):
            if not(np.array_equal(original,other)):
                RMSE,MSE,PSNR,SSIM = 0,0,0,0
                for k in range(3): #3 Channels/image --> Grayscale good enough
                    RMSE += calcRMSE(original[:,:,k], other[:,:,k]) #Create agreggate
                    MSE += calcMSE(original[:,:,k], other[:,:,k]) #Create agreggate
                    PSNR += calcPSNR(original[:,:,k], other[:,:,k]) #Create agreggate
                    SSIM += calcSSIM(original[:,:,k], other[:,:,k]) #Create agreggate
                RMSE = RMSE / 3
                MSE = MSE/ 3
                PSNR = PSNR/ 3
                SSIM = SSIM/ 3
                print("Comparing Image " + str(i) + " and " + str(j))
                print([RMSE,MSE,PSNR,SSIM])
                values.append([RMSE,MSE,PSNR,SSIM])
    else: #Do nothing
        pass

#Plot all images
fig, ax = plt.subplots(ncols = images.shape[0])
ax= ax.ravel()
label = 'MSE: {:.2f}, SSIM: {:.2f}, RMSE: {:.2}, PSNR: {:.2f}'
for i in range(images.shape[0]):
    ax[i].imshow(images[i])
    ax[i].axis("off")
    #ax[i].set_xlabel(label.format(values[i], values))
    ax[i].set_title('Image '+str(i))
plt.show()

