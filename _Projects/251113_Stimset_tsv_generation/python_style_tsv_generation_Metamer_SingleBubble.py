'''
This script will generate tsv info for stimsets we used here.

tsv in sequence:
ID,FileName,Stim_Type,Category,Raw_Graph

Category is divided by '_', e.g. P4_C1

'''
#%%

import OS_Tools as ot
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#%%
############################ P1-Mega_Metamer_v251104 #############################

# load ML tsv file
filename = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Mega_Metamer_v250920','.tsv')[0]
stim_infos = pd.read_csv(filename, sep='\t')

# add columns for annotate.
stim_infos['Stim_Set']='Default' # stim set of given stim
stim_infos['Category']='Default' # category of stimtype,sepereted by '_'
stim_infos['Object']=-1 # object of given stim, 1-40 as input data sets, and -1 for fob.

#%% generate fob 150*2 parts
# fob parts
stim_sets = []
categories = []
objects = []
for i in range(2): # 2 repeats
    for i in range(15):
        categories.append('Face')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Body')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Object')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Scene')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Food')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Face_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Body_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Object_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Scene_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Food_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)

#%% 1-series, ordinary metamer parts

for i in range(5): # cycle repeats
    for j in range(5): # cycle constrains
        for k in range(40): # cycle objects
            stim_sets.append('Metamer')
            objects.append(k+1) # 1-40
            if j == 0:
                if k<20:
                    categories.append('Raw_Raw_Ani')
                else:
                    categories.append('Raw_Raw_Inani')
            elif j==1:
                if k<20:
                    categories.append('P4_C4_Ani')
                else:
                    categories.append('P4_C4_Inani')
            elif j==2:
                if k<20:
                    categories.append('P4_C3_Ani')
                else:
                    categories.append('P4_C3_Inani')
            elif j==3:
                if k<20:
                    categories.append('P4_C2_Ani')
                else:
                    categories.append('P4_C2_Inani')
            elif j==4:
                if k<20:
                    categories.append('P4_C1_Ani')
                else:
                    categories.append('P4_C1_Inani')
#%% 2-series, color,ani only.

colors = ['Gray_Ani','Rev_Ani','Red_Ani','Green_Ani','Blue_Ani']
for i in range(5): # cycle colors
    for j in range(20): # cycle objects
        categories.append(colors[i])
        stim_sets.append('Color_Ver')
        objects.append(j+1)

#%% 3-series, silct and boulder
for i in range(20):
    categories.append('Boulder')
    stim_sets.append('Boulder_Silct')
    objects.append(i+1)

for i in range(20):
    categories.append('Silct')
    stim_sets.append('Boulder_Silct')
    objects.append(i+1)

#%% 5-series, cut-shuffle, with spatial constrained.
cut_methods = ['Cut12_C4_S3','Cut12_C3_S4','Cut12_C2_S6','Cut12_C1_S12','Cut8_C4_S2','Cut9_C3_S3','Cut8_C2_S4','Cut8_C1_S8','Cut4_C1_S4','Cut4_C2_S2'] # c meaning constrain,shuffle meaning shuffle inside constrained parts

for i in range(2): # 2 cycles
    for j in range(10): # 10 cut methods
        for k in range(40): # 40 different objects
            stim_sets.append('Cut_Shuffle')
            categories.append(cut_methods[j])
            objects.append(k+1)

#%% At last, combine all this into tsv file.
stim_infos['Category'] = categories
stim_infos['Stim_Set'] = stim_sets
stim_infos['Object'] = objects

stim_infos.to_csv('Mega_Metamer_v250920.tsv', sep='\t', index=True)

#%% No mask required here.
# SKIP Mask Generation, Use try-expect method for info loader.
