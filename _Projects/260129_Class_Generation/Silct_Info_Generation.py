'''

生成新的silct数据集对应的info文件，参照metamer的格式。

'''


#%%

import OS_Tools as ot
import csv
import pandas as pd
import matplotlib.pyplot as plt


#%%
############################ Doodle_AI_v260119 and  Doodle_AI_v260121 #############################

filename = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Doodle_AI_v260119','.tsv')[0]
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

#%% 1-3,original doodles
# 1 Series, doodle
for i in range(400): # cycle repeats
    stim_sets.append('Doodles')
    objects.append(i+1) # 1-400
    categories.append('Doodle')

# 2 Series, doodle-boulder
for i in range(400): # cycle repeats
    stim_sets.append('Doodles')
    objects.append(i+1) # 1-400
    categories.append('Boulder')

# 3 Series, doodle-boulder
for i in range(400): # cycle repeats
    stim_sets.append('Doodles')
    objects.append(i+1) # 1-400
    categories.append('Silct')

#%% 4 series,3 set of AI shape-reserved.
for j in range(3):
    for i in range(400):
        stim_sets.append('Doodles')
        objects.append(i+1) # 1-400
        categories.append('Shape_Reserved_AI')
#%% 5 series, 2 set of AI symantic generated.
for j in range(2):
    for i in range(400):
        stim_sets.append('Doodles')
        objects.append(i+1) # 1-400
        categories.append('Generated_AI_Category')

#%% At last, combine all this into tsv file.
stim_infos['Category'] = categories
stim_infos['Stim_Set'] = stim_sets
stim_infos['Object'] = objects

stim_infos.to_csv('Doodle_AI_v260119.tsv', sep='\t', index=True)
