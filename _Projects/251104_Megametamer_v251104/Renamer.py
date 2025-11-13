'''
This stim set actually merge metamer and cut-shuffle altogether.

So based on metamercut and megametamer 0920, we make this stim set.

'''
#%%

import OS_Tools as ot
from tqdm import tqdm
import os 

wp=r'C:\#working_folder\#Codes\ZR_Blab_Base\_Projects\251104_Megametamer_v251104\500'
all_names = ot.Get_File_Name(wp,'.jpg')

#%% rename 
for i,c_name in enumerate(all_names):
    c_name_parts = c_name.split('\\')
    c_name_parts[-1] = '7'+c_name_parts[-1][1:]
    after_name = '\\'.join(c_name_parts)
    os.rename(c_name, after_name)