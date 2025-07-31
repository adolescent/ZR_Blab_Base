'''
These functions will operate matrix, getting the result in need.
'''
import numpy as np
from sklearn.decomposition import PCA


def Matrix_Clipper(matrix,percent = 99.5):# how much you want to remain unchanged.
    flattened = matrix.flatten()
    boulder = (100-percent)/2
    lower_bound = np.percentile(flattened,boulder)
    upper_bound = np.percentile(flattened,100-boulder)
    clipped_matrix = np.clip(matrix, lower_bound, upper_bound)

    return clipped_matrix


def Get_Extreme(matrix,method = 'max',axis = 1,prop=0.05):
    '''
    Calculate a matrix's max/min value toward an axis,
    WORKS ONLY ON 2D MATRIX!!

    '''
    if axis == 0:# transfer 
        matrix = matrix.T

    n_cols = matrix.shape[1]
    k = max(1, int(n_cols * prop))  
    if method == 'max':
        partitioned = np.partition(matrix, -k, axis=1)
        values = partitioned[:, -k:]
    elif method == 'min':
        partitioned = np.partition(matrix, k-1, axis=1)
        values = partitioned[:, :k]
    else:
        raise ValueError('Invalid Extreme Method.')

    return values.mean(axis=1)


def Corr_Matrix(data,fill_diag=True):
    """
    计算样本间的皮尔逊相关系数矩阵并置对角线为0
    
    参数:
    data -- 形状为(d, n)的NumPy数组，d为特征维度，n为样本数量
            本例中d=250, n=1416
    
    返回:
    corr_matrix -- 形状为(n, n)的皮尔逊相关矩阵，对角线为0
    """
    # 转置数据使每行为一个样本 (n x d)
    samples = data.T
    
    # 计算均值
    mean = np.mean(samples, axis=1, keepdims=True)
    
    # 中心化数据
    centered = samples - mean
    
    # 计算标准差 (添加小值避免除零错误)
    std_dev = np.sqrt(np.sum(centered**2, axis=1))
    std_dev[std_dev == 0] = 1e-10
    
    # 计算协方差矩阵
    cov_matrix = np.dot(centered, centered.T)
    
    # 计算皮尔逊相关系数矩阵
    corr_matrix = cov_matrix / (std_dev[:, np.newaxis] * std_dev)
    
    # 将对角线置为0
    if fill_diag:

        np.fill_diagonal(corr_matrix, 0)
    
    return corr_matrix


def Do_PCA(Z_frame,feature = 'Cell',pcnum = 20,method = 'full'):
    # NOTE: INPUT FRAME SHALL BE IN SHAPE N_Cell*N_Img
    # feature can be 'Cell' or 'Img', indicating which axis is feature and which is sample.
    pca = PCA(n_components = pcnum,svd_solver=method) # random
    data = np.array(Z_frame)
    if feature == 'Cell':
        data = data.T# Use cell as sample and frame as feature.
    elif feature == 'Img':
        data = data
    else:
        raise ValueError('Sample method invalid.')
    pca.fit(data)
    PC_Comps = pca.components_# out n_comp*n_feature
    point_coords = pca.transform(data)# in n_sample*n_feature,out n_sample*n_comp
    return PC_Comps,point_coords,pca


