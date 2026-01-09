'''
汇总全部的神经响应,并存成一个dict

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

dcnn_path = r'E:\#Preprocessed_Data\251230_Metamer_DCNN_Response'
neuro_path = r'E:\#Preprocessed_Data\Selected_Cells'
# used_time = [160,320]

#%%
## 汇总时统一了顺序，排列方式均为N_img*N_dim,且只保留了一千张图片

All_Resp = {}
All_Resp['Alex_conv']=np.load(ot.Join(dcnn_path,'alexnet_features.npy'))[:1000,:]
All_Resp['VGG19_conv']=np.load(ot.Join(dcnn_path,'vgg19_last_conv_features.npy'))[:1000,:]
All_Resp['Res18_conv']=np.load(ot.Join(dcnn_path,'resnet18_layer4_4_features.npy'))[:1000,:]
All_Resp['Alex_fc']=np.load(ot.Join(dcnn_path,'alexnet_fc7_features.npy'))[:1000,:]
All_Resp['VGG19_fc']=np.load(ot.Join(dcnn_path,'vgg19_fc2_features.npy'))[:1000,:]
All_Resp['Res18_avgpool']=np.load(ot.Join(dcnn_path,'res18_pool_features.npy'))[:1000,:]



#%% 读取神经数据
All_Resp['MSB'] = np.load(ot.Join(neuro_path,'MSB_Cells_Metamer_Only.npz'))['psth'][:,:,160:320].sum(-1).T
All_Resp['ASB'] = np.load(ot.Join(neuro_path,'ASB_Cells_Metamer_Only.npz'))['psth'][:,:,160:320].sum(-1).T
All_Resp['MF'] = np.load(ot.Join(neuro_path,'MF_Cells_Metamer_Only.npz'))['psth'][:,:,160:320].sum(-1).T
All_Resp['AL'] = np.load(ot.Join(neuro_path,'AL_Cells_Metamer_Only.npz'))['psth'][:,:,160:320].sum(-1).T

#%% 保存成一个dict，使用pickle
ot.Save_Variable(dcnn_path,'All_Response',All_Resp)


