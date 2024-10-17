import numpy as np
import time
import glob 
import cv2
import os
 
from sklearn.model_selection import train_test_split
from skimage.color import rgb2grey

files = []
labels  = []

directory = 'train'
list_of_files = os.listdir(directory)
for file in list_of_files:
    name = os.path.splitext(os.path.basename(file))
    subfiles = os.listdir(directory+'/'+name)
    for sub in subfiles:
        files.append(directory+'/'+name+'/'+sub)
        labels.append(name)
	print(directory+'/'+name+'/'+sub)


