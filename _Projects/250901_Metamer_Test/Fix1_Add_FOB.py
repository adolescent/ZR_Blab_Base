'''
We need to add fob after 1k stimset, making it possible for cell selection.

'''

#%%

import numpy
import OS_Tools as ot
from tqdm import tqdm
import shutil



fob_path = r'D:\#stimuli\STI_150'
fig_path = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\Metamer_Pool4_C4321_Object'

savepath = r'D:\#stimuli\Metamer_P4_C4321_Object_STI150_1300'

n_repeat = 2

#%%
# get fob and real data path.
fig_pts = ot.Get_File_Name(fig_path,'.jpg')
fob_pts = ot.Get_File_Name(fob_path,'.jpg')

counter = 1
for i,c_file in tqdm(enumerate(fig_pts)):
    c_name = str(10000+counter)[1:]+'.jpg'
    tar_path = ot.Join(savepath,c_name)
    shutil.copy2(c_file, tar_path)
    counter +=1

for n in range(n_repeat):
    for j,c_fob in tqdm(enumerate(fob_pts)):
        c_name = str(10000+counter)[1:]+'.jpg'
        tar_path = ot.Join(savepath,c_name)
        shutil.copy2(c_fob, tar_path)
        counter +=1