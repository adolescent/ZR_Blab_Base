'''
这个脚本用来处理文章中的结果:
比较三选一的ODD-1 detection任务,
处理类似原始文本中的三分类任务,根据神经表征挑选三分类的正确率,以及比较vgg 和alexnet fc6 进行表征区分的正确率。
------
有两种方式：
- 原始 vs 打乱
- 原始图A vs 原始图B
------
使用的是皮尔逊距离:sqrt(1-r^2),计算每个和剩下两个的平均距离然后做softmax,方法非常简单。


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


raw_path = r'E:\#Stimsets\Raw_Objects'
raw_names = ot.Get_File_Name(raw_path)

dcnn_resp_path = r'E:\#Preprocessed_Data\Metamer_DCNN_Response'
alex_conv=np.load(ot.Join(dcnn_resp_path,'alexnet_features.npy'))
vgg_conv=np.load(ot.Join(dcnn_resp_path,'vgg19_last_conv_features.npy'))
res_conv=np.load(ot.Join(dcnn_resp_path,'resnet18_layer4_4_features.npy'))
# alex_conv=np.load(ot.Join(dcnn_resp_path,'alexnet_fc7_features.npy'))
# vgg_conv=np.load(ot.Join(dcnn_resp_path,'vgg19_fc2_features.npy'))
# res_conv=np.load(ot.Join(dcnn_resp_path,'res18_pool_features.npy'))

#%%################################# Run parts #################################
# for all 
from scipy.special import softmax
from itertools import combinations
all_names = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Metamer_P4_C4321_Object_STI150_1300')
all_names.sort()

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
            raw_alex = alex_conv[graph_id,:]
            c4_alex = alex_conv[c_pair[0],:]
            c42_alex = alex_conv[c_pair[1],:]
            raw_vgg = vgg_conv[graph_id,:]
            c4_vgg = vgg_conv[c_pair[0],:]
            c42_vgg = vgg_conv[c_pair[1],:]            
            raw_res = res_conv[graph_id,:]
            c4_res = res_conv[c_pair[0],:]
            c42_res = res_conv[c_pair[1],:]
            
            # alexnet
            a,_ = pearsonr(raw_alex,c4_alex)
            b,_ = pearsonr(raw_alex,c42_alex)
            c,_ = pearsonr(c4_alex,c42_alex)
            # a = np.sqrt(1-a**2)
            # b = np.sqrt(1-b**2)
            # c = np.sqrt(1-c**2)
            a = 1-a
            b = 1-b
            c = 1-c
            c_correct = softmax([(a+b)/2,(a+c)/2,(b+c)/2])[0]
            correct_rate.loc[counter,:] = [c_constrain,j,c_correct,'Alexnet']
            counter += 1 
            # vgg
            a_vgg,_ = pearsonr(raw_vgg,c4_vgg)
            b_vgg,_ = pearsonr(raw_vgg,c42_vgg)
            c_vgg,_ = pearsonr(c4_vgg,c42_vgg)
            a_vgg = 1-a_vgg
            b_vgg = 1-b_vgg
            c_vgg = 1-c_vgg
            c_correct_vgg = softmax([(a_vgg+b_vgg)/2,(a_vgg+c_vgg)/2,(b_vgg+c_vgg)/2])[0]
            correct_rate.loc[counter,:] = [c_constrain,j,c_correct_vgg,'VGG16']
            counter += 1 
            # resnet
            a_res,_ = pearsonr(raw_res,c4_res)
            b_res,_ = pearsonr(raw_res,c42_res)
            c_res,_ = pearsonr(c4_res,c42_res)
            a_res = 1-a_res
            b_res = 1-b_res
            c_res = 1-c_res
            c_correct_res = softmax([(a_res+b_res)/2,(a_res+c_res)/2,(b_res+c_res)/2])[0]
            correct_rate.loc[counter,:] = [c_constrain,j,c_correct_res,'Resnet50']
            counter += 1 

correct_rate = correct_rate.dropna(how='any')
# correct_rate = correct_rate.astype('f8')
#%% Plot generated graph.

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,5),dpi=240)
correct_rate.Constrain = correct_rate.Constrain.astype('str')
sns.lineplot(data=correct_rate,x ='Constrain',y='Prop_Correct',ax=ax,errorbar='ci',hue='Network',legend=False)
sns.boxplot(data=correct_rate,x ='Constrain',y='Prop_Correct',ax=ax,hue='Network',width=0.3, showfliers=False)

ax.axhline(1/3,linestyle='--',color='gray')
ax.set_ylim(0,1)
ax.set_xticklabels([4,3,2,1])
ax.set_ylabel('Correct Prop.')

correct_rate.to_csv('Network_Correct_Rate_last_conv.csv')
#%% load in real neuron data, and calculate it's odd detection effects.
neu_folder = r'E:\#Preprocessed_Data\Selected_Cells'


