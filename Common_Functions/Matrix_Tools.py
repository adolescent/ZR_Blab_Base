'''
These functions will operate matrix, getting the result in need.
'''
import numpy as np



def Matrix_Clipper(matrix,percent = 99.5):# how much you want to remain unchanged.
    flattened = matrix.flatten()
    boulder = (100-percent)/2
    lower_bound = np.percentile(flattened,boulder)
    upper_bound = np.percentile(flattened,100-boulder)
    clipped_matrix = np.clip(matrix, lower_bound, upper_bound)

    return clipped_matrix