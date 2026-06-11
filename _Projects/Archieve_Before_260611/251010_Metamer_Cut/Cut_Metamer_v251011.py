'''
Generate Stimsets.

'''




#%%

import OS_Tools as ot
from tqdm import tqdm
import numpy as np
import copy
from PIL import Image
import random
import shutil

savepath=r'D:\#stimuli\Metamer_Cut_v251011\Metamer_Cut_v251013'


'''
FOB and metamer use original 00000 and 10000 series 
silct and boulder as 30000 series.
'''
octo_path=r'D:\#stimuli\Metamer_Cut_v251011\Octo_cut'
n_path=r'D:\#stimuli\Metamer_Cut_v251011\N_cut'



#%% OCTO shuffle as 40000 series.
counter=1
octo_names = ot.Get_File_Name(octo_path,'.jpg')
for i,c_name in enumerate(octo_names):
        tar_name = str(40000+counter)+'.jpg'
        tar_path = ot.Join(savepath,tar_name)
        shutil.copy2(c_name, tar_path)
        counter +=1

#%% N shuffle as 50000 series.

counter=1
n_names = ot.Get_File_Name(n_path,'.jpg')
for i,c_name in enumerate(n_names):
        tar_name = str(50000+counter)+'.jpg'
        tar_path = ot.Join(savepath,tar_name)
        shutil.copy2(c_name, tar_path)
        counter +=1



