import numpy as np
import cv2
import os
from image_registration import chi2_shift
from scipy.ndimage import shift
from matplotlib import pyplot as plt

# Path to the images
path_images = os.path.expanduser('~') + '\Pictures\Stitching_images'

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Reading all the images
images = []
if os.path.isdir(path_images):
    entries = os.listdir(path_images)
    i = 0
    printProgressBar(i, len(entries), prefix = 'Reading images:', suffix = 'Complete', length = 100) 
    for entry in entries:
        images.append(cv2.imread(path_images + '\\' + entry))        
        i += 1
        printProgressBar(i, len(entries), prefix = 'Reading images:', suffix = 'Complete', length = 100)