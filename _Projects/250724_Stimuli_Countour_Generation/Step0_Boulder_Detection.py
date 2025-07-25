'''
This script will transfer data into boulder lines.
After boulderization, will graph be possible for line-detection.


Try canny algorithms for boulder detection?
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, color
from Stimuli_Function.Graph_Trans import *
import cv2
import random


image = io.imread(r'D:\#Data\#stimuli\FOB96\FOB2023short\047.png')
gray_image = color.rgb2gray(image)*255
blurred = cv2.GaussianBlur(gray_image, (11,11),5)
boulders = Boulder_Canny((blurred).astype('u1'),20,25)
plt.imshow(boulders)


