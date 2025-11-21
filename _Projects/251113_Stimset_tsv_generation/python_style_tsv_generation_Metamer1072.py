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



#%%
############################ P1-Mega_Metamer_v251104 #############################

# load ML tsv file
filename = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\_Recycle\Metamer_Pool4_C4321_Object_1k+FOB','.tsv')[0]
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
for i in range(1): # 1 repeats
    for i in range(24):
        categories.append('Body')
        stim_sets.append('FOB_FOB72')
        objects.append(-1)
    for i in range(24):
        categories.append('Face')
        stim_sets.append('FOB_FOB72')
        objects.append(-1)
    for i in range(24):
        categories.append('Object')
        stim_sets.append('FOB_FOB72')
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


#%% 3-series, silct and boulder
stim_infos['Category'] = categories
stim_infos['Stim_Set'] = stim_sets
stim_infos['Object'] = objects

stim_infos.to_csv('Metamer1072.tsv', sep='\t', index=True)
#%% 
