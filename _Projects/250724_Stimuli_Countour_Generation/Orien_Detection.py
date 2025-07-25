'''
This script will get overall orientation of a given graph.
'''


#%% Use functions to do this work.
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

from Stimuli_Function.Graph_Trans import *

image = io.imread(r'D:\#Data\silct\Dataset\silct_npx_1416\0142.jpg')
gray_image = color.rgb2gray(image)


magnitude_spectrum = FFT_2D(gray_image,mag_thres=200)
# plt.imshow(magnitude_spectrum,cmap='gray')
angle_deg,_ = Spectrum_Angle_Estimation(magnitude_spectrum)




