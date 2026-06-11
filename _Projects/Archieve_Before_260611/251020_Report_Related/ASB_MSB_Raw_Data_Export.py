



#%%

import matplotlib.pyplot as plt
import numpy as np
import OS_Tools as ot
import seaborn as sns
import pandas as pd
from tqdm import tqdm

msb_psth = np.load(r'D:\_DataTemp\Metamer\PSTH\250828_JianJian_Metamer_Pool4_C4321_Object_1k+FOB_g6_AL_ASB_PSTH.npy')

asb_psth = np.load(r'D:\_DataTemp\Metamer\PSTH\250902_ZhuangZhuang_Metamer_P4_C4321_Object_STI150_1300_g2_MSB_PSTH.npy')


#%%
msb_psth_on = msb_psth[:,:,:,160:320].sum(-1)
asb_psth_on = asb_psth[:,:,:,160:320].sum(-1)


#%%
from scipy import io

io.savemat('msb_metamer_raw_bytrail.mat', {'MSB_Response': msb_psth_on})
io.savemat('asb_metamer_raw_bytrail.mat', {'ASB_Response': asb_psth_on})
