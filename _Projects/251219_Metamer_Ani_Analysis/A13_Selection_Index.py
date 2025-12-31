'''
参照文献中方法, 计算选择性index(real-syn)/(syna-synb)

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

