'''
This script will generate countours on a given sized graph, making it 

'''
#%%
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

# Initialize a 400x400 matrix with zeros (black background)
matrix_size = 400
image_matrix = np.zeros((matrix_size, matrix_size), dtype=np.uint8)

# Grid size and bar length
grid_size = 25
bar_length = 10  # Fixed length for all bars

# Define possible angles (0, 22.5, ..., 337.5 degrees)
angles = [i * 20 for i in range(18)]

# Draw random bars in each segment
for i in range(16):
    for j in range(16):
        # Calculate the center of the segment
        center_x = (i * grid_size) + (grid_size // 2)
        center_y = (j * grid_size) + (grid_size // 2)

        # Random angle
        angle = random.choice(angles)

        # Calculate the end coordinates of the bar
        rad_angle = np.deg2rad(angle)
        start_x = int(center_x - (bar_length / 2) * np.cos(rad_angle))
        start_y = int(center_y - (bar_length / 2) * np.sin(rad_angle))
        end_x = int(center_x + (bar_length / 2) * np.cos(rad_angle))
        end_y = int(center_y + (bar_length / 2) * np.sin(rad_angle))
        
        # Draw the bar (line) on the matrix
        cv2.line(image_matrix, (start_x, start_y), (end_x, end_y), 255, 2)

# Now image_matrix is a 400x400 uint8 matrix with white bars on a black background
plt.imshow(image_matrix)