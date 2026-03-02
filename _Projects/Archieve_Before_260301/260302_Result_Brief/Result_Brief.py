'''
简要描述到目前为止的一些结果，并为进一步的工作做准备。
'''

#%%

import numpy as np
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
import pandas as pd
from tqdm import tqdm


wp=r'E:\#Preprocessed_Data\Selected_Cells'
cellname = 'Doodle_All_MSB_Response.npz'
fullpath = ot.Join(wp,cellname)

ac_infos = np.load(fullpath,allow_pickle=True)
ac_dp = pd.DataFrame(ac_infos['d_primes'],columns = ['Cell','DP','Category','Loc','Stimset','ID'])
ac_rsp = pd.DataFrame(ac_infos['response'],columns = ['Cell','DP','Category','Loc','Stimset','ID'])
#%% get cell counts
cell_num,_,_ = ac_infos['psth'].shape
body_dp = ac_dp[ac_dp.Category=='Body'].reset_index(drop=True)
cell_num = len(body_dp)
set(body_dp.Loc)
#%%
counter = 0
for i in tqdm(range(cell_num)):
    ccloc = body_dp.loc[i,'Loc']
    # if ('Maodan' in ccloc) or ('MD' in ccloc) or ('MaoDan' in ccloc):
    if 'Mao' in ccloc:
        counter +=1
