'''
This script will syth graph with 12 different parameters, 3 for each.
'''
#%%
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from syn_ai import FeatureSynthesizer
import OS_Tools as ot
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

save_path = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\Pool4_C4321_Object_only_Repeat5'
raw_path = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\Pool4_Shuffle_Object_Only'
lr=0.05
N_iter = 15000

all_img_name = ot.Get_File_Name(raw_path,'.jpg')

#%%
counter=0
N_repeat = 5
constrain_set = [(4,4),(3,3),(2,2),(1,1)]
# pool_set = [['pool1','pool2','pool4'],['pool1','pool2'],['pool1']]
pool_set = [['pool1','pool2','pool4']]


# generate all graphs 
counter=0
for n in tqdm(range(N_repeat)):# repeat all data set 3 times
    
    # raw as first 80
    for i,c_path in enumerate(all_img_name):
        c_img_name = c_path.split('\\')[-1][:-4]
        c_img = Image.open(c_path).convert("RGB")
        c_filename = str(10001+counter)[1:]
        c_img.save(ot.Join(save_path,c_filename+'.jpg'))
        counter +=1


    # then repeat constrain and pool,sequence is C4P4,C4P2,C4P1,C3P4,C3P2,C3P1...
    for i,c_constrain in enumerate(constrain_set):
        print(f'Current_constrain:{c_constrain}')
        for j,c_pool in enumerate(pool_set):
            print(f'Current_pool:{c_pool}')
            for k,c_path in enumerate(all_img_name):
                c_img_name = c_path.split('\\')[-1][:-4]
                c_img = Image.open(c_path).convert("RGB")
                Synther = FeatureSynthesizer(
                c_img, 
                spatial_constraint=c_constrain,  # 全局约束 (完全打乱)
                feature_layers=c_pool,
                num_steps=N_iter,step_mute = True,learning_rate=lr
                )
                synth_img = Synther.synthesize()
                # resive img into 400x400
                synth_img_resized =  synth_img.resize((400, 400))

                c_filename = str(10001+counter)[1:]
                synth_img_resized.save(ot.Join(save_path,c_filename+'.jpg'))
                counter +=1






