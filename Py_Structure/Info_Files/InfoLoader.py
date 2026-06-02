'''

Including FOB file of catagory info, for tuning calculation.

'''

#%%
import pandas as pd
import OS_Tools as ot
import numpy as np
import os

#%%

def Load_Info(setname='Metamer_Singlebubble_v251107',load_mask=False):
    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(current_file_path)
    # print(root_dir)
    c_tsv_path = setname+'.tsv'
    c_mask_path = 'Masks_'+setname+'.npz'
    real_tsv_path = ot.Join(root_dir,c_tsv_path)
    real_mask_path = ot.Join(root_dir,c_mask_path)
    

    try:
        tsv_info = pd.read_csv(real_tsv_path, sep='\t')
    except FileNotFoundError:
        print('Info not recored, check name plz.')
        tsv_info = None

    if load_mask == False:
        masks = None
        raw_mask_file = None
    else:
        try:
            raw_mask_file = np.load(real_mask_path)
            masks = raw_mask_file['masks']
        except FileNotFoundError:
            print('No mask found.')
            raw_mask_file = None
            masks = None

    return tsv_info,masks,raw_mask_file

#%% Define a new function, include how we select stim ids of fob and real response.
# UPDATE EACH TIME YOU HAVE NEW STIMSET.

def Select_Cell_Info(stim_info='Anagram'):

    select_dicts = {}
    if stim_info == 'Anagram':
        stim_sets = ['Anagram_Jigsaw_v260227']
        fob_styles = ['STI150']
        fobids = [np.arange(150)]# if multi times, just use them all. 
        data_id_list = [np.arange(150, 750)]
    elif stim_info == 'Doodle':
        pass
        stim_sets = ['Doodle_AI_v260119','Doodle_AI_v260121','Doodle_AI_v260430']
        fob_styles = ['STI150','STI150','Wordloc']
        fobids = [np.arange(300),np.arange(300),np.arange(180)]
        data_id_list = [np.arange(300, 3500),np.arange(300, 3500),np.arange(180, 3380)]

    for i, c_stimset in enumerate(stim_sets):
        select_dicts[c_stimset] = {
            'FOB': {'style': fob_styles[i], 'id': fobids[i]},
            'Data': data_id_list[i],
        }
    return select_dicts


#%% test run
if __name__ == '__main__':

    name = 'Metamer_Singlebubble_v251107'
    a,b,c = Load_Info(setname=name,load_mask=True)

    