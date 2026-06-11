'''

Preproess 1, transfer Goodunit Data into PSTH matrix, using default onset time -100~350ms

Stimulus onset is 250on+150off in 10 degrees.

'''


#%%
import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np


wp=r'D:\#Data\Metamer'
savepath=r'D:\#Data\Metamer\PSTH'

gn_names = ot.Get_File_Name(wp,'.mat')

#%%
for i,c_gn in tqdm(enumerate(gn_names)):
    c_name = c_gn.split('\\')[-1][9:-4]
    c_psth = PSTH_From_Goodunit(c_gn,img_num=1072)
    np.save(ot.Join(savepath,c_name+"_PSTH"),c_psth)



