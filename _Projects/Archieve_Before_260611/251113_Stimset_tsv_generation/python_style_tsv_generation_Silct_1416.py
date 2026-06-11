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
filename = ot.Get_File_Name(r'Z:\Monkey\Stimuli\LXY\silct_npx_1416','.tsv')[0]
stim_infos = pd.read_csv(filename, sep='\t')

# add columns for annotate.
stim_infos['Stim_Set']='Default' # stim set of given stim
stim_infos['Category']='Default' # category of stimtype,sepereted by '_'
stim_infos['Object']=-1 # object of given stim, 1-40 as input data sets, and -1 for fob.

#%% Silct stimuli, 400 objects, in sequence texture-boulder-silct.
stim_sets = []
categories = []
objects = []

cats = ['Texture','Boulder','Silct']
for i in range(400): # cycle objects
    for j in range(3): # cycle type
        categories.append(cats[j])
        stim_sets.append('Texture_Boulder_Silct')
        objects.append(i+1)



#%% fob parts at last

for i in range(3): # 1 repeats
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
    
#%% 3-series, silct and boulder
stim_infos['Category'] = categories
stim_infos['Stim_Set'] = stim_sets
stim_infos['Object'] = objects

stim_infos.to_csv('Silct1416.tsv', sep='\t', index=True)
#%% 
