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

filename = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\260227_Anagram_Jigsaw','.tsv')[0]
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
for i in range(1): # 2 repeats
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

#%% 1 A-B set of anagram jigsaw
# 1 Series, doodle
for i in range(3): # Cycle Repeats
    for j in range(20): # Cycle obj pairs
        for k in range(5): # Cycle Styles
            c_pair_name= f'Obj{j+1}_Style{k+1}_Repeat{i+1}'
            # Graph A
            categories.append(c_pair_name+'_A')
            stim_sets.append('Anagram_Jigsaw')
            objects.append(j+1)
            # Graph B
            categories.append(c_pair_name+'_B')
            stim_sets.append('Anagram_Jigsaw')
            objects.append(j+1+20)


#%% At last, combine all this into tsv file.
stim_infos['Category'] = categories
stim_infos['Stim_Set'] = stim_sets
stim_infos['Object'] = objects

stim_infos.to_csv('Anagram_Jigsaw_v260227.tsv', sep='\t', index=True)
