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

savepath=r'D:\#stimuli\Metamer_Mega_v250918\Mega_Metamer_v250920'


fob_path = r'D:\#stimuli\STI_150'
meta_path=r'D:\#stimuli\Metamer_P4_C4321_Object_STI150_1300'
color_rev_path=r'D:\#stimuli\Metamer_Mega_v250918\Base40_gimp_trans'
boulder_path=r'D:\#stimuli\Metamer_Mega_v250918\Base40_gimp_trans'
shuffle_path = r'D:\#stimuli\Metamer_Mega_v250918\Base40_Cut'

#%% copy target graph into path.
counter=1
fob_files = ot.Get_File_Name(fob_path,'.jpg')
for N in range(2):
    for i,c_name in enumerate(fob_files):

        tar_name = str(100000+counter)[1:]+'.jpg'
        tar_path = ot.Join(savepath,tar_name)
        shutil.copy2(c_name, tar_path)
        counter +=1

#%% then copy memamer data into it.
metamer_files = ot.Get_File_Name(meta_path,'.jpg')[:1000]
counter=1
for i,c_name in enumerate(metamer_files):
        tar_name = str(10000+counter)+'.jpg'
        tar_path = ot.Join(savepath,tar_name)
        shutil.copy2(c_name, tar_path)
        counter +=1
#%% then add color rev path.
color_names = []
color_names.extend(ot.Get_File_Name(color_rev_path,'.jpg','gray'))
color_names.extend(ot.Get_File_Name(color_rev_path,'.jpg','rev'))
color_names.extend(ot.Get_File_Name(color_rev_path,'.jpg','R'))
color_names.extend(ot.Get_File_Name(color_rev_path,'.jpg','G'))
color_names.extend(ot.Get_File_Name(color_rev_path,'.jpg','B'))

counter=1
for i,c_name in enumerate(color_names):
        tar_name = str(20000+counter)+'.jpg'
        tar_path = ot.Join(savepath,tar_name)
        shutil.copy2(c_name, tar_path)
        counter +=1
#%% boulder and silct
boulder_names = []
boulder_names.extend(ot.Get_File_Name(color_rev_path,'.jpg','boulder'))
boulder_names.extend(ot.Get_File_Name(color_rev_path,'.jpg','silct'))
counter=1
for i,c_name in enumerate(boulder_names):
        tar_name = str(30000+counter)+'.jpg'
        tar_path = ot.Join(savepath,tar_name)
        shutil.copy2(c_name, tar_path)
        counter +=1
#%% shuffle_path
shuffle_path = ot.Get_File_Name(shuffle_path)
for i,c_name in enumerate(shuffle_path):
        tar_name = str(50000+counter)+'.jpg'
        tar_path = ot.Join(savepath,tar_name)
        shutil.copy2(c_name, tar_path)
        counter +=1
