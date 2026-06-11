'''
This script will assemble stimulus set and 

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
import OS_Tools as ot
from tqdm import tqdm
import os
from syn_ai import FeatureSynthesizer

wp = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\raw\cropped'
save_path = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\real_stim'

#%% generate tsv and save 
all_img_name = ot.Get_File_Name(wp,'.jpg')

counter=0
for i,c_path in enumerate(all_img_name):
    c_img_name = c_path.split('\\')[-1][:-4]
    c_img = Image.open(c_path).convert("RGB")
    c_filename = str(10001+counter)[1:]
    c_img.save(ot.Join(save_path,c_filename+'.jpg'))
    counter +=1

#%% then add (4,4),(3,3),(2,2),(1,1) for graph syn.
constrain_set = [(4,4),(3,3),(2,2),(1,1)]

for j,c_constrain in enumerate(constrain_set):
    print(f'Current_constrain:{c_constrain}')
    for i,c_path in tqdm(enumerate(all_img_name)):

        c_img_name = c_path.split('\\')[-1][:-4]
        c_img = Image.open(c_path).convert("RGB")

        Synther = FeatureSynthesizer(
            c_img, 
            spatial_constraint=c_constrain,  # 全局约束 (完全打乱)
            feature_layers=['pool1','pool2','pool4'],
            num_steps=10000,step_mute = True
        )
        synth_img = Synther.synthesize()
        # resive img into 400x400
        synth_img_resized =  synth_img.resize((400, 400))

        c_filename = str(10001+counter)[1:]
        synth_img_resized.save(ot.Join(save_path,c_filename+'.jpg'))
        counter +=1


