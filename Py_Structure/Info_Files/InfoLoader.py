'''

Including FOB file of catagory info, for tuning calculation.

'''

#%%
import pandas as pd
import OS_Tools as ot
import numpy as np

#%%

def Load_Info(setname='Metamer_Singlebubble_v251107',load_mask=False):
    try:
        tsv_info = pd.read_csv(setname+'.tsv', sep='\t')
    except FileNotFoundError:
        print('Info not recored, check name plz.')
        tsv_info = None

    if load_mask == False:
        masks = None
        raw_mask_file = None
    else:
        try:
            raw_mask_file = np.load('Masks_'+setname+'.npz')
            masks = raw_mask_file['masks']
        except FileNotFoundError:
            print('No mask found.')
            raw_mask_file = None
            masks = None

    return tsv_info,masks,raw_mask_file




#%% test run
if __name__ == '__main__':

    name = 'Metamer1300wesf'
    a,b,c = Load_Info(setname=name,load_mask=True)