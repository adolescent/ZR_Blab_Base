'''
读取各个脑区的响应并进行Odd-1 detection 任务, 对照dCNN的正确率。

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

neuro_path = r'E:\#Preprocessed_Data\Selected_Cells'
msb_cells = np.load(ot.Join(neuro_path,'MF_Cells_Metamer_Only.npz'),allow_pickle=True)
msb_resps = msb_cells['psth'][:,:,160:320].sum(-1)

#%%

scale=10

correct_rate = pd.DataFrame(index=range(100000),columns=['Constrain','Graph','Prop_Correct','Network'])


# graph_id = 1
# c_level = 1
# data = [graph_id+40*c_level,graph_id+200+40*c_level,graph_id+400+40*c_level,graph_id+600+40*c_level,graph_id+800+40*c_level]
# all_pairs = list(combinations(data, 2))

counter=0
for l in range(1,5):
    c_level = l
    c_constrain = 5-l
    print(f'Current Constrain:C{c_constrain}')
    for j in tqdm(range(20)):
        graph_id =j
        data = [graph_id+40*c_level,graph_id+200+40*c_level,graph_id+400+40*c_level,graph_id+600+40*c_level,graph_id+800+40*c_level]
        all_pairs = list(combinations(data, 2))
        # print(f'Current ID: {graph_id}')
        for i,c_pair in enumerate(all_pairs):
            raw_alex = msb_resps[:,graph_id]
            c4_alex = msb_resps[:,c_pair[0]]
            c42_alex = msb_resps[:,c_pair[1]]

            # alexnet
            # a,_ = pearsonr(raw_alex,c4_alex)
            # b,_ = pearsonr(raw_alex,c42_alex)
            # c,_ = pearsonr(c4_alex,c42_alex)
            # # a = np.sqrt(1-a**2)
            # # b = np.sqrt(1-b**2)
            # # c = np.sqrt(1-c**2)
            # a = 1-a
            # b = 1-b
            # c = 1-c
            # ## 欧氏距离的话？
            a = np.sqrt(np.sum((raw_alex - c4_alex)**2))
            b = np.sqrt(np.sum((raw_alex - c42_alex)**2))
            c = np.sqrt(np.sum((c4_alex - c42_alex)**2))
            dists = np.array([(a+b)/2,(a+c)/2,(b+c)/2])

            c_correct = softmax(dists/dists.sum())[0]
            # c_correct = dists[0]/dists.sum()
            correct_rate.loc[counter,:] = [c_constrain,j,c_correct,'ASB_Normed_Raw']
            counter += 1 

            c_correct = softmax(dists*scale/dists.sum())[0]
            # c_correct = dists[0]/dists.sum()
            correct_rate.loc[counter,:] = [c_constrain,j,c_correct,f'ASB_Scale{scale}']
            counter += 1 


correct_rate = correct_rate.dropna(how='any')
# correct_rate = correct_rate.astype('f8')
#%% Plot generated graph.

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,5),dpi=240)
correct_rate.Constrain = correct_rate.Constrain.astype('str')
sns.lineplot(data=correct_rate,x ='Constrain',y='Prop_Correct',ax=ax,errorbar='ci',hue='Network')
# sns.boxplot(data=correct_rate,x ='Constrain',y='Prop_Correct',ax=ax,hue='Network',width=0.3, showfliers=False)

ax.axhline(1/3,linestyle='--',color='gray')
ax.set_ylim(0.3,0.5)
ax.set_xticklabels([4,3,2,1])
ax.set_ylabel('Correct Prop.')

# correct_rate.to_csv('ASB_Correct_Rate.csv')
#%% #############读取所有csv并绘图###########
net_corr_conv = pd.read_csv('Network_Correct_Rate_last_conv.csv', index_col=0)
suffix = "_Conv"
net_corr_conv = net_corr_conv.map(lambda x: f"{x}{suffix}" if isinstance(x, str) else x)

net_corr_fc = pd.read_csv('Network_Correct_Rate_last_fc.csv', index_col=0)
suffix = "_FC"
net_corr_fc = net_corr_fc.map(lambda x: f"{x}{suffix}" if isinstance(x, str) else x)

msb_corr = pd.read_csv('MSB_Correct_Rate.csv', index_col=0)
asb_corr = pd.read_csv('ASB_Correct_Rate.csv', index_col=0)
mf_corr = pd.read_csv('MF_Correct_Rate.csv', index_col=0)
al_corr = pd.read_csv('AL_Correct_Rate.csv', index_col=0)

all_corrs = pd.concat([net_corr_conv,net_corr_fc,msb_corr,asb_corr,mf_corr,al_corr]).reset_index(drop=True)
#%%

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8),dpi=240)
all_corrs.Constrain = all_corrs.Constrain.astype('str')
# sns.lineplot(data=all_corrs,x ='Constrain',y='Prop_Correct',ax=ax,errorbar='ci',hue='Network',)
sns.boxplot(data=all_corrs,x ='Constrain',y='Prop_Correct',ax=ax,hue='Network',width=0.3, showfliers=False)

ax.axhline(1/3,linestyle='--',color='gray')
ax.set_ylim(0.3,0.4)
ax.set_xticklabels([4,3,2,1])
ax.set_ylabel('Correct Prop.')
