'''
This script will get overall orientation of a given graph.
'''

#%%

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# Load the image
image = io.imread(r'D:\#Data\silct\Dataset\silct_npx_1416\0142.jpg')  # Replace with your image path
mag_thres = 200

gray_image = color.rgb2gray(image)

# Compute the 2D FFT
f_transform = np.fft.fft2(gray_image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Compute the magnitude spectrum,this orientation is orthogonal to original graph.
magnitude_spectrum = np.abs(f_transform_shifted)
magnitude_spectrum = np.clip(magnitude_spectrum,0,mag_thres)

# then we need to get the major 














#%% BACKUP
# Compute the angles of the FFT components
y_indices, x_indices = np.indices(magnitude_spectrum.shape)
angles = np.arctan2(y_indices - magnitude_spectrum.shape[0] // 2, 
                     x_indices - magnitude_spectrum.shape[1] // 2)

# Sum the magnitude spectrum along the radial direction
hist, bin_edges = np.histogram(angles.flatten(), bins=180, weights=magnitude_spectrum.flatten())

# Find the orientation with the maximum value
dominant_orientation = np.degrees(bin_edges[np.argmax(hist)])

# Plotting the original image and the orientation bar
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.bar(0, max(hist), orientation='vertical', width=0.1, color='blue', alpha=0.7)
plt.xlim(-90, 90)
plt.ylim(0, max(hist))
plt.title('Dominant Orientation')
plt.axvline(dominant_orientation, color='red', linestyle='--', label=f'Orientation: {dominant_orientation:.2f}°')
plt.legend()
plt.xlabel('Angle (degrees)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
