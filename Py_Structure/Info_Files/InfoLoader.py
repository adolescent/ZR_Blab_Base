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




#%% test run
if __name__ == '__main__':

    name = 'Metamer_Singlebubble_v251107'
    a,b,c = Load_Info(setname=name,load_mask=True)