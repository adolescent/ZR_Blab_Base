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
filename = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Metamer_P4_C4321_Object_STI150_1300','.tsv')[0]
stim_infos = pd.read_csv(filename, sep='\t')

# add columns for annotate.
stim_infos['Stim_Set']='Default' # stim set of given stim
stim_infos['Category']='Default' # category of stimtype,sepereted by '_'
stim_infos['Object']=-1 # object of given stim, 1-40 as input data sets, and -1 for fob.

#%% generate fob 150*2 parts
# fob parts at last!
stim_sets = []
categories = []
objects = []

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
#%% FOB part at last = =
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


#%% save parts
stim_infos['Category'] = categories
stim_infos['Stim_Set'] = stim_sets
stim_infos['Object'] = objects

stim_infos.to_csv('Metamer1300.tsv', sep='\t', index=True)
#%% no mask required
