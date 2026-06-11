'''
参照文献中方法, 计算选择性index(real-syn)/(syna-synb)
同时也可以, 得到每张图, 每个打乱对每个网络的三角形, 识别的等变性？

这个script还是使用原始的皮尔逊距离

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
from itertools import combinations
from scipy.spatial import distance
import matplotlib.pyplot as plt

neuro_path = r'E:\#Preprocessed_Data\Selected_Cells'
all_rsps = ot.Load_Variable(r'E:\#Preprocessed_Data\251230_Metamer_DCNN_Response','All_Response.pkl')
all_nets = list(all_rsps.keys())

#%%## 计算选择概率,dist 用pearson R
all_dists = pd.DataFrame(index=range(1000000),columns=['Network','Graph','Metric','Constrain','Raw_C1','Raw_C2','CC'])
counter = 0

# Odd category detection prop对照在一边
for i,cnet in enumerate(all_nets):# cycle networks
    print(f'Current Network:{cnet}')
    cc_rsp = all_rsps[cnet]
    # cc_rsp = cc_rsp/cc_rsp.max(0,keepdims=True)
    # cc_rsp = np.nan_to_num(cc_rsp) ## Normalization 不改变结果。
    for j in tqdm(range(40)):# cycle imgs
        for k in range(1,5):# cycle constrains
            c_constrain = 5-k
            data = [j+40*k,j+200+40*k,j+400+40*k,j+600+40*k,j+800+40*k]
            all_pairs = list(combinations(data, 2)) # shuffle pairs
            for l,cc_pair in enumerate(all_pairs):
                cc_raw = cc_rsp[[j,j+200,j+400,j+600,j+800],:].mean(0)
                cc_m1 = cc_rsp[cc_pair[0],:]
                cc_m2 = cc_rsp[cc_pair[1],:]
                # 计算皮尔逊距离
                r1,_ = pearsonr(cc_raw,cc_m1)
                r2,_ = pearsonr(cc_raw,cc_m2)
                r3,_ = pearsonr(cc_m1,cc_m2)
                d1_p = 1-r1
                d2_p = 1-r2
                d3_p = 1-r3
                # 计算欧氏距离
                d1_e = distance.euclidean(cc_raw,cc_m1)
                d2_e = distance.euclidean(cc_raw,cc_m2)
                d3_e = distance.euclidean(cc_m1,cc_m2)
                all_dists.loc[counter,:] = [cnet,j,'Pearson_Dist',c_constrain,d1_p,d2_p,d3_p]
                counter += 1
                all_dists.loc[counter,:] = [cnet,j,'Euclidean_Dist',c_constrain,d1_e,d2_e,d3_e]
                counter += 1

all_dists = all_dists.dropna(how='any')


#%% 计算Odd 类别识别, 即对不同图-同一级别的打乱, odd1 out的识别概率
for i in range(len(all_nets)):
    subset = all_dists.query(f"Network == '{all_nets[i]}' & Metric == 'Pearson_Dist' & Constrain == 1 & Graph == 8")
    dist_raw_syn = (subset['Raw_C1'].mean()+subset['Raw_C2'].mean())/2
    # print(subset['Raw_C2'].mean())
    dist_syns = subset['CC'].mean()
    index = (dist_raw_syn-dist_syns)/(dist_raw_syn+dist_syns)

    print(f'Current Network:{all_nets[i]},Index:{index:.4f}')
    print(dist_raw_syn)
    print(dist_syns)
    
#%% 三角形绘制
# A为CC距离，BC为两条Raw-Constrain，ABC顺时针。
def draw_upward_triangle(a, b, c,ax,color ='#2c3e50' ):
    # 1. 验证三角形不等式
    if (a + b <= c) or (a + c <= b) or (b + c <= a):
        print("错误：边长无法组成三角形")
        return

    # 2. 计算顶点 A 处的内角 alpha (对应边 a 的对角)
    cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
    # 限制范围防止浮点数精度问题导致的 nan
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    # 3. 设置顶点坐标
    # A 点固定
    A = np.array([0.5, 0])
    
    # B 点在 A 的左上方
    B = np.array([
        A[0] - c * np.sin(alpha / 2),
        A[1] + c * np.cos(alpha / 2)
    ])
    
    # C 点在 A 的右上方
    C = np.array([
        A[0] + b * np.sin(alpha / 2),
        A[1] + b * np.cos(alpha / 2)
    ])

    # 4. 绘图准备
    points = np.array([A, B, C, A])
    
    # plt.figure(figsize=(7, 7))
    ax.plot(points[:, 0], points[:, 1], '-', color=color, linewidth=2)
    # plt.fill(points[:, 0], points[:, 1], color='orange', alpha=0.2)

    # 5. 辅助线：绘制过 A 的垂直平分参考线
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, label='Symmetry Axis')

    # 标注点
    # plt.text(A[0], A[1] - 0.02, 'A (0.5, 0)', ha='center', fontweight='bold')
    # plt.text(B[0], B[1] + 0.01, f'B (side c={c:.2f})', ha='right')
    # plt.text(C[0], C[1] + 0.01, f'C (side b={b:.2f})', ha='left')

    # 设置样式
    ax.axis('equal')
    # ax.set_ylim(-0.05, max(B[1], C[1]) + 0.1)
    # plt.title(f"Upward Triangle (Sides: a={a}, b={b}, c={c})")
    ax.grid(False, alpha=0.2)
    # ax.show()


# draw_upward_triangle(a=0.2, b=0.3, c=0.35)
#%% example, 展示对一组metamer打乱各个网络的识别能力
example = all_dists.query(f"Network == 'Alex_conv' & Metric == 'Pearson_Dist' & Constrain == 1 & Graph == 8").reset_index(drop=True)
a,b,c = example.loc[0,['CC','Raw_C1','Raw_C2']]
sum = a+b+c


fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(7,7))
draw_upward_triangle(a/sum,b/sum,c/sum,ax,color='#2c3e50')

example2 = all_dists.query(f"Network == 'Alex_fc' & Metric == 'Pearson_Dist' & Constrain == 1 & Graph == 8").reset_index(drop=True)
a,b,c = example2.loc[0,['CC','Raw_C1','Raw_C2']]
sum = a+b+c
draw_upward_triangle(a/sum,b/sum,c/sum,ax,color="#6ca3db")

# draw_upward_triangle(example.loc[])

#%% 平均一个网络对所有图片的响应，并绘制等腰的对比三角形

## alex conv
example_conv = all_dists.query(f"Network == 'Res18_conv' & Metric == 'Pearson_Dist' & Constrain == 1").reset_index(drop=True)

a = example_conv.CC.mean()
b = example_conv.Raw_C1.mean()
c = example_conv.Raw_C2.mean()
# 等腰化
bc = (b+c)/2
sum = a+2*bc

# 绘图
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(7,7))
draw_upward_triangle(a/sum,bc/sum,bc/sum,ax,color='#2c3e50')

## alex fc
example_conv = all_dists.query(f"Network == 'Res18_avgpool' & Metric == 'Pearson_Dist' & Constrain == 1").reset_index(drop=True)

a = example_conv.CC.mean()
b = example_conv.Raw_C1.mean()
c = example_conv.Raw_C2.mean()
# 等腰化
bc = (b+c)/2
sum = a+2*bc

# 绘图
draw_upward_triangle(a/sum,bc/sum,bc/sum,ax,color='#6ca3db')


#%% ########################### 计算 selection index，并统计 ##############################
all_dists['Selection_Index'] = 99999.00
for i in tqdm(range(len(all_dists))):
    cc_a = all_dists.loc[i,'Raw_C1']
    cc_b = all_dists.loc[i,'Raw_C2']
    cc_between = all_dists.loc[i,'CC']
    cc_raw_syn = (cc_a+cc_b)/2
    cc_index = (cc_raw_syn-cc_between)/(cc_raw_syn+cc_between)
    all_dists.loc[i,'Selection_Index'] = cc_index
#%% 
plotable = all_dists.query(f"Metric == 'Pearson_Dist'").reset_index(drop=True)

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(12,5))
sns.barplot(data=plotable,x='Network',y='Selection_Index',ax=ax,hue='Constrain',width=0.5,palette='tab10')
ax.set_ylim(-0.2,1)


#%% Image-by-Image的分类正确率，还是用原始的odd 1 out方法，clust by img，横轴是dcnn(alex)，纵轴是脑区(msb) 
net_data = all_dists.query(f"Network == 'Alex_conv' & Metric == 'Pearson_Dist' & Constrain == 1").reset_index(drop=True)
neuro_data = all_dists.query(f"Network == 'MSB' & Metric == 'Pearson_Dist' & Constrain == 1").reset_index(drop=True)
concats = pd.concat([neuro_data,net_data]).reset_index(drop=True)
concats['Ani'] = np.where(concats['Graph'] < 20, 'Ani', 'Inani')
concats['Corr_Rate']=0.0
for i in tqdm(range(len(concats))):
    c_slide = concats.iloc[i,:]
    cc_a = c_slide['Raw_C1']
    cc_b = c_slide['Raw_C2']
    cc_c = c_slide['CC']
    dists = [(cc_a+cc_b)/2,(cc_a+cc_c)/2,(cc_b+cc_c)/2]
    c_corr = softmax(dists)[0]
    concats.iloc[i,-1]=c_corr

# plot parts
plotable = pd.DataFrame(columns=['MSB','Alex_conv','Graph','Ani'],index = range(400))
for i in range(400):
    plotable.loc[i,:] = [concats.loc[i,'Selection_Index'],concats.loc[i+400,'Selection_Index'],concats.loc[i,'Graph'],concats.loc[i,'Ani']]

# plotable = concats.pivot_table(columns=['Network'],values='Selection_Index',index='Graph')
# plotable['Ani'] = np.where(plotable.index < 20, 'Ani', 'Inani')

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(5,5))
sns.scatterplot(data=plotable,x='Alex_conv',y='MSB',ax=ax,lw=0,s=10,hue='Ani')
# ax.axhline(y=1/3,linestyle='--',color='gray',alpha=0.5)
# ax.axvline(x=1/3,linestyle='--',color='gray',alpha=0.5)
ax.plot([-0.2,1],[-0.2,1],linestyle='--',color='gray',alpha=0.5)
# ax.set_ylim(-0.2,1)
ax.set_ylim(-0.2,1)
ax.set_xlim(-0.2,1)


