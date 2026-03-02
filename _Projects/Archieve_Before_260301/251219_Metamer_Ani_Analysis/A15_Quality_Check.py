'''

对结果进行质量检查,包括响应强度-相似性矩阵等等

'''

#%%
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os 
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
from tqdm import tqdm
from scipy.stats import pearsonr
import numpy as np
from scipy.special import softmax
from itertools import combinations,permutations
from scipy.spatial import distance
import matplotlib.pyplot as plt

neuro_path = r'E:\#Preprocessed_Data\Selected_Cells'
all_rsps = ot.Load_Variable(r'E:\#Preprocessed_Data\251230_Metamer_DCNN_Response','All_Response.pkl')
all_nets = list(all_rsps.keys())
#%%  # 展示各个网络的内部情况，发现20的循环只在脑区中真实存在，而网络没有这种特性
fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(10,5),dpi=300)

sns.heatmap(all_rsps['MSB'][:200,:],center=0,vmax=5,cbar=False,xticklabels=False,yticklabels=False,ax=ax[0],cmap='bwr')
sns.heatmap(all_rsps['Alex_conv'][:200,:],center=0,vmax=5,cbar=False,xticklabels=False,yticklabels=False,ax=ax[1],cmap='bwr')
sns.heatmap(all_rsps['Alex_fc'][:200,:],center=0,cbar=False,xticklabels=False,yticklabels=False,ax=ax[2],cmap='bwr')
ax[0].set_title('MSB',fontsize=8)
ax[1].set_title('Alex conv',fontsize=8)
ax[2].set_title('Alex fc7',fontsize=8)



#%% ### 检查A，响应强度，包括对每张图的平均相应强度，以及每个cell的响应强度。

# 数据集生成
data_list = []
for cnet, cc_response in all_rsps.items():
    print(f'Current Network: {cnet}')
    n_img, n_cell = cc_response.shape
    
    # 1. 利用向量化生成 j (img_id 的索引)
    j_indices = np.arange(n_img)
    
    # 2. 向量化计算 img_id, c_constrain, c_img
    img_ids = j_indices % 200
    c_constrains = 5 - (img_ids // 40)
    c_imgs = img_ids % 40
    
    # 3. 构造该 Network 下的所有组合 (Cartesian Product)
    # 使用 repeat 将图片信息扩展，以匹配所有的 cell
    # j_expanded 形状为 (n_img * n_cell,)
    net_col = np.array([cnet] * (n_img * n_cell))
    cell_col = np.tile(np.arange(n_cell), n_img)
    
    img_col = np.repeat(c_imgs, n_cell)
    constrain_col = np.repeat(c_constrains, n_cell)
    
    # Normalize,得到当前活动的归一化值
    cc_response = cc_response/cc_response.max(0,keepdims=True)
    cc_response = np.nan_to_num(cc_response)
    # 4. 拉平 response 矩阵 (n_img, n_cell) -> (n_img * n_cell,)
    resp_col = cc_response.flatten()
    
    # 5. 合并为临时 DataFrame 或 Dictionary
    temp_df = pd.DataFrame({
        'Network': net_col,
        'Cell': cell_col,
        'Img': img_col,
        'Constrain': constrain_col,
        'Response': resp_col
    })
    data_list.append(temp_df)

# 最后一次性合并所有结果
all_rsps_frame = pd.concat(data_list, ignore_index=True)
all_rsps_frame.to_csv('All_Response_Frame.csv')
#%% 不用这个方法了
# for i,cnet in enumerate(all_nets):
#     # cnet_rsps = all_rsps_frame.query("Network == {cnet}")
#     cnet_rsps = all_rsps_frame[all_rsps_frame.Network==cnet]

#     # 我们先创建一个辅助序列，只包含 Constrain 为 5 的 Value，其他为 NaN
#     cnet_rsps['baseline'] = cnet_rsps['Response'].where(cnet_rsps['Constrain'] == 5)
#     # 使用 groupby + transform('mean') 将该均值广播（Broadcast）到每个分组的所有行
#     # 这样每一行都会对应其所属 (Cell, Img) 组中 Constrain 5 的平均值 (好聪明！)
#     group_baselines = cnet_rsps.groupby(['Cell', 'Img'])['baseline'].transform('mean')
#     # 除以每个反应的baseline，得到当前活动和raw的比值
#     cnet_rsps['Normalized_Response'] = cnet_rsps['Response'] / group_baselines
# a = all_rsps_frame.groupby('Network').get_group('MSB')
# sns.boxplot(data = a.groupby('Img').get_group(0),x='Constrain',y='Response',showfliers=False)
brain_areas = all_nets[6:]
network_convs = all_nets[:3]
network_fcs = all_nets[3:6]

fig,ax = plt.subplots(nrows=1,ncols=4,dpi=240,figsize=(7,4),sharex=True,sharey=True)
for i,cnet in tqdm(enumerate(brain_areas)):
    cc_response = all_rsps_frame[all_rsps_frame.Network==cnet]
    # cc_response = cc_response[cc_response.Img<20]
    cc_response['Constrain_str'] = cc_response['Constrain'].astype(str)
    my_order = ['5', '4', '3', '2', '1']
    my_labels = ['Raw', 'C4', 'C3', 'C2', 'C1']

    sns.boxplot(data=cc_response,x='Constrain',y='Response',ax = ax[i],width=0.5,showfliers=False,zorder=1,order=my_order)
    sns.pointplot(data=cc_response, x='Constrain', y='Response', ax=ax[i], color='black', order=my_order, errorbar='ci', markers='', linestyles='-', zorder=2,lw=1)

    # ax[i%2,i//2].set_xticks([5,4,3,2,1])
    ax[i].set_xticklabels(my_labels)
    ax[i].set_title(f'{cnet}',fontsize=8)

    


#%% ### 检查B，获得相似性矩阵，各个网络的，包括图片别的和图片平均的。
# 三组，图片别，ani平均，全部平均
RSM = {} # 全部的相似性矩阵

#### 生成新的id
n_groups = 5  # 5组
n_treatments = 5  # 5种处理
n_images_per_treatment = 40  # 每种处理40张
total_images = n_groups * n_treatments * n_images_per_treatment  # 1000
# 生成新索引
new_indices = []
for img_idx in range(n_images_per_treatment):
# 遍历5种处理
    for treatment_idx in range(n_treatments):
        # 遍历5个组
        for group_idx in range(n_groups):
            # 计算原始索引
            # 组偏移: group_idx * 200 (每组200张: 5种处理 × 40张)
            # 处理偏移: treatment_idx * 40
            # 图片偏移: img_idx
            original_idx = group_idx * 200 + treatment_idx * 40 + img_idx
            new_indices.append(original_idx)

# 将列表转换为numpy数组
new_indices = np.array(new_indices)

for i,cnet in enumerate(all_nets):
    cc_response = all_rsps[cnet]
    cc_response = cc_response/cc_response.max(0,keepdims=True) # normalize, if needed
    cc_response = np.nan_to_num(cc_response)
    arranged_response = cc_response[new_indices,:]
    c_rsm = np.corrcoef(arranged_response)
    # c_rsm = np.nan_to_num(c_rsm)
    RSM[cnet] = c_rsm
    # sns.heatmap(c_rsm,center=0,cmap='bwr')

#%% Plot parts
brain_areas = all_nets[6:]
network_convs = all_nets[:3]
network_fcs = all_nets[3:6]
fig,ax = plt.subplots(nrows=1,ncols=3,dpi=240,figsize=(11,4),sharex=True,sharey=True)
for i,cnet in tqdm(enumerate(network_convs)):
    sns.heatmap(RSM[cnet],center=0,cmap='bwr',square=True,ax=ax[i],xticklabels=False,yticklabels=False,cbar=False)
    ax[i].set_title(cnet,fontsize=8)
fig.tight_layout()

#%%





