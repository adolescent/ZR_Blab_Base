'''
确认图片没问题, 不同类别的图片进行odd 1 out 识别的分辨能力。

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

#%% 建立一个对照表，
all_dists = pd.DataFrame(index=range(1000000),columns=['Network','Img_A','Img_B','Constrain','Raw_C1','Raw_C2','CC'])

counter = 0
# Odd category detection prop对照在一边
for i,cnet in enumerate(all_nets):# cycle networks
    cc_rsp = all_rsps[cnet]
    print(f'Current Network:{cnet}')
    # 循环图片对,得到不同匹配
    all_imgs = list(range(20)) # 只选择animate对照
    all_img_pairs = list(permutations(all_imgs, 2)) # 使用组合，AB倒过来
    for j,c_img_pair in tqdm(enumerate(all_img_pairs)):# 循环图片对
        ca_raw = c_img_pair[0]
        cb_raw = c_img_pair[1]
        for k in range(1,5): #循环constrain
            c_constrain = 5-k
            # a 和B的任意两张组合做corr
            a_sets = [ca_raw+k*40,ca_raw+k*40+200,ca_raw+k*40+400,ca_raw+k*40+600,ca_raw+k*40+800]
            b_sets = [cb_raw+k*40,cb_raw+k*40+200,cb_raw+k*40+400,cb_raw+k*40+600,cb_raw+k*40+800]
            all_b_pairs = list(combinations(b_sets, 2)) # shuffle pairs
            for l,c_b_pair in enumerate(all_b_pairs): # 循环全部B对
                for m,c_a_set in enumerate(a_sets):

                    cc_raw = cc_rsp[c_a_set,:]
                    cc_m1 = cc_rsp[c_b_pair[0],:]
                    cc_m2 = cc_rsp[c_b_pair[1],:]
                    r1,_ = pearsonr(cc_raw,cc_m1)
                    r2,_ = pearsonr(cc_raw,cc_m2)
                    r3,_ = pearsonr(cc_m1,cc_m2)
                    d1_p = 1-r1
                    d2_p = 1-r2
                    d3_p = 1-r3
                    all_dists.loc[counter,:] = [cnet,ca_raw,cb_raw,c_constrain,d1_p,d2_p,d3_p]
                    counter += 1

all_dists = all_dists.dropna(how='any')

#%% 计算Odd 类别识别, 即对不同图-同一级别的打乱, odd1 out的识别概率
all_dists['Selection_Index'] = 99999.00
for i in tqdm(range(len(all_dists))):
    cc_a = all_dists.loc[i,'Raw_C1']
    cc_b = all_dists.loc[i,'Raw_C2']
    cc_between = all_dists.loc[i,'CC']
    cc_raw_syn = (cc_a+cc_b)/2
    cc_index = (cc_raw_syn-cc_between)/(cc_raw_syn+cc_between)
    all_dists.loc[i,'Selection_Index'] = cc_index
#%%
# plotable = all_dists.query(f" Constrain == 4").reset_index(drop=True)
plotable = all_dists

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(12,5))
sns.barplot(data=plotable,x='Network',y='Selection_Index',ax=ax,hue='Constrain',width=0.5,palette='tab10')
ax.set_ylim(-0.2,1)

