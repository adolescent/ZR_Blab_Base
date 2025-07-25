'''
These functions are used for graph transformation.
Changing it will help us getting right orietation of graphs

'''

#%%


import numpy as np
import cv2


def FFT_2D(gray_image,mag_thres = 200):
    '''
    Input gray-scale graph, and return 2D magnify img. This will help elavulate info of input graphs
    mag_thres : threshold
    '''
    # Compute the 2D FFT
    height,width = gray_image.shape
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum,this orientation is orthogonal to original graph.
    magnitude_spectrum = np.abs(f_transform_shifted)
    # remove center part and clip graph for better estimation
    magnitude_spectrum[height//2,width//2]=0
    magnitude_spectrum = np.clip(magnitude_spectrum,0,mag_thres)

    return magnitude_spectrum


def Spectrum_Angle_Estimation(density_map):
    '''
    Estimate angle of magnitude spectrum. This function will be useful for duplicate graph with lines into it.
    '''

    # 1. 创建坐标网格
    height, width = density_map.shape
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    # 2. 展平坐标和密度值
    coordinates = np.vstack([xx.ravel(), yy.ravel()]).T
    weights = density_map.ravel()

    # 3. 计算加权均值（中心点）
    weighted_sum = np.sum(weights)
    mean_x = np.sum(xx.ravel() * weights) / weighted_sum
    mean_y = np.sum(yy.ravel() * weights) / weighted_sum

    # 4. 中心化坐标
    centered_coords = coordinates - [mean_x, mean_y]

    # 5. 计算加权协方差矩阵
    cov_matrix = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            cov_matrix[i, j] = np.sum(centered_coords[:, i] * centered_coords[:, j] * weights) / weighted_sum

    # 6. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 7. 获取主成分方向（最大方差方向）
    primary_idx = np.argmax(eigenvalues)
    primary_vector = eigenvectors[:, primary_idx]

    # 8. 计算斜率（注意图像坐标系中Y轴向下）
    # 在数学坐标系中，斜率 = dy/dx
    # 但在图像坐标系中，Y轴方向相反，因此需要取负
    # slope = -primary_vector[1] / primary_vector[0]

    # 9. 计算角度（相对于水平轴）
    angle_rad = np.arctan2(primary_vector[1], primary_vector[0])
    angle_deg = np.degrees(angle_rad)

    # 确保角度在-90到90度之间
    if angle_deg < -90:
        angle_deg += 180
    elif angle_deg > 90:
        angle_deg -= 180

    return angle_deg,angle_rad

def Img_Center(img):
    height, width = img.shape
    y_indices, x_indices = np.indices((height, width))
    total_weight = np.sum(img)
    # 计算加权平均坐标
    centroid_x = np.sum(x_indices * img) / total_weight
    centroid_y = np.sum(y_indices * img) / total_weight
    return centroid_x,centroid_y



def Boulder_Canny(gray, canny_low=3, canny_high=50):

    # Reduce noise with Gaussian blur
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = gray
    # Detect edges using Canny
    edges = cv2.Canny(blurred, canny_low, canny_high)

    return edges.astype('f8')
