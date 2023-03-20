import cv2
import os
import numpy as np

path ='/ssd3/sunting/PP-Industry/dataset/MCITF/SD-saliency-900/Ground truth/'
for im in os.listdir(path):


    image = cv2.imread(path + im)
    print(np.unique(image))
