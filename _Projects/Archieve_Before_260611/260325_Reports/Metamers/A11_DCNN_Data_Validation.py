'''
使用DCNN和原文中的方法来验证,确实我们的metamer对network来说是无法识别的
方法是画三角形，两两配对，验证


同时注意三角形的大小，即pearsonR的绝对值变化。

这个要保存一个data frame，是每个network特征层，对每个配对的距离-相似性。

'''


#%%
import seaborn as sns
import OS_Tools as ot
from PIL import Image
import numpy as np
from Matrix_Tools import Corr_Matrix
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

datafolder=r'E:\#Preprocessed_Data\Selected_Cells'
# filename = r'Res50_Response.npz'
# filename = r'Alex_Response.npz'
filename = r'VGG16_Response.npz'

savepath = r'E:\#Preprocessed_Data\260305_Report_Data'

data = np.load(ot.Join(datafolder,filename))
keys = list(data.keys())
n_img = 1000
#%%

img_counter = 0
for i,c_layer in tqdm(enumerate(keys)):
    c_response = data[c_layer].reshape(n_img,-1)
    data_reshaped = c_response.reshape(25, 40, c_response.shape[1])

    # 2. 遍历每一张原图 (0-39)
    for img_idx in range(40):
        img_data = data_reshaped[:,img_idx,:]  # 形状为 (25, 4096)
        # 计算这25个变体之间的相关系数矩阵 (25x25)
        # np.corrcoef 会计算行与行之间的相关性
        corr_matrix = np.corrcoef(img_data)
        results = []
        # 3. 获取上三角阵的索引（排除自相关，避免重复计算 A-B 和 B-A）
        rows, cols = np.triu_indices(25, k=1)
        
        for r, c in zip(rows, cols):
            results.append({
                'Network':'VGG16',
                'Layer': c_layer,

                'Img_Index': img_idx,
                'C_img1': r,  # 变体1的编号 (0-24)
                'C_img2': c,  # 变体2的编号 (0-24)
                'Corr': corr_matrix[r, c],
                'Dist': 1-corr_matrix[r, c]
            })
        df = pd.DataFrame(results)

        # 4. 转化为 DataFrame
        if img_counter == 0:
            Network_Corr = copy.deepcopy(df)
        else:
            Network_Corr = pd.concat([Network_Corr,df])
        img_counter+=1
# save pd frame.
Network_Corr.to_parquet(ot.Join(savepath,'VGG16_Corr.parquet'))

#%% ################## Plot 1, CORR MATRIX  ####################
plotable = Corr_Matrix(data['last_conv'].reshape(1000,-1).T,fill_diag=False)[:200,:200]

fig,ax = plt.subplots(ncols=1,nrows=1,dpi = 240,figsize=(5,5))
sns.heatmap(plotable,ax=ax,cbar=False,square=True,xticklabels=False,yticklabels=False,cmap='bwr',center=0)



#%% ################## Plot 2, Graph_AVR_Constrain_Corr  ####################
# get mirror network and concat it.
# 1. 确保镜像对称
plotable = Network_Corr[Network_Corr.Layer=='fc1']

df_mirror = plotable.copy()
# 假设你的列名是 C_img1, C_img2, Corr
df_mirror.columns = ['Network','Layer', 'Img_Index', 'C_img2', 'C_img1','Corr','Dist']
df_total = pd.concat([plotable, df_mirror], axis=0)

df_total['C_img1'] = df_total['C_img1']%5
df_total['C_img2'] = df_total['C_img2']%5

# 2. 使用 pivot_table 代替 pivot
# index: 行坐标, columns: 列坐标, values: 填充值
# aggfunc='mean': 核心点！它会自动处理所有重复索引并取平均
matrix = df_total.pivot_table(
    index='C_img1', 
    columns='C_img2', 
    values='Corr', 
    aggfunc='mean'
)
#
# 3. 绘图
fig,ax = plt.subplots(ncols=1,nrows=1,dpi = 240,figsize=(5,5))
sns.heatmap(matrix, annot=False, cmap='RdBu_r', center=0,vmax =1,cbar=False,xticklabels=False,yticklabels=False,square=True,ax=ax)
ax.set_xlabel('')
ax.set_ylabel('')

# plt.title('Averaged Correlation Matrix')
plt.show()


