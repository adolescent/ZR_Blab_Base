'''
This will plot redplots for 

'''


#%%
import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np



path=r'D:\#Data\Metamer\PSTH'

psth_files = ot.Get_File_Name(path,'.npy')

exp_psth = np.load(psth_files[1])


#%% odd-end ceiling 
'''
Try to do noise ceiling using fob part of data, Use odd&end similarity>0.3 as good cell.


'''
ceiling_index = odd_end_ceiling(exp_psth[:,:,:72,:],np.arange(150,250))
ok_cells = np.where(ceiling_index>0.3)[0]
ceiled_response = exp_psth.mean(1)[ok_cells,:,:]

redplot = Redplot(ceiled_response,np.arange(75,125),np.arange(150,250))


#%% get tuning of cells.
fob_index = pd.DataFrame('',index=np.arange(72),columns=['Category'])
for i in range(72):
    if i <24:
        c_category = 'Body'
    elif i<48:
        c_category = 'Face'
    else:
        c_category = 'Object'
    fob_index.loc[i] = c_category

tuning_frame = Calculate_Cell_Tunings(ceiled_response[:,:72,:],fob_index)



