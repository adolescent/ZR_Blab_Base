'''
Export data from npy to mat all cell response.

The cat is 0003,0323,0923
'''

#%%

import matplotlib.pyplot as plt
import numpy as np
import OS_Tools as ot
import seaborn as sns
import pandas as pd
from tqdm import tqdm


msb_resp = ot.Load_Variable(r'D:\_DataTemp\Metamer\ceiled_response\MSB\250902_ZhuangZhuang_Metamer_P4_C4321_Object_STI150_1300_g2_MSB_PSTH_Ceiled.pkl')[0]
asb_resp = ot.Load_Variable(r'D:\_DataTemp\Metamer\ceiled_response\ASB\250828_JianJian_Metamer_Pool4_C4321_Object_1k+FOB_g6_AL_ASB_PSTH_Ceiled.pkl')[0]

# msb_index = np.where(msb_resp[0]>0.5)[0]
# asb_index = np.where(asb_resp[0]>0.5)[0]
# msb_resp = msb_resp[1][msb_index,:]
# asb_resp = asb_resp[1][asb_index,:]
msb_resp = msb_resp[:,:1000,160:320].mean(-1)
asb_resp = asb_resp[:,72:,160:320].mean(-1)


#%%
from scipy import io

io.savemat('msb_metamer_raw.mat', {'MSB_Response': msb_resp})
io.savemat('asb_metamer_raw.mat', {'ASB_Response': asb_resp})

#%%
from Matrix_Tools import Corr_Matrix
a = Corr_Matrix(asb_resp[:,:200],fill_diag=False)
sns.heatmap(a,center=0,vmax = 1)