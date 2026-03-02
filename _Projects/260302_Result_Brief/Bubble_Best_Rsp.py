'''
比较每个神经元，最好的bubble反应占原始图片翻译的比例。

'''

#%%
import numpy as np
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

wp=r'E:\#Preprocessed_Data\Selected_Cells'
cellname = 'MSB_Cells_Bubble.npz'
fullpath = ot.Join(wp,cellname)

ac_infos = np.load(fullpath,allow_pickle=True)
ac_dp = pd.DataFrame(ac_infos['d_primes'],columns = ['Cell','DP','Category','Loc','ID'])
ac_rsp = pd.DataFrame(ac_infos['response'],columns = ['Cell','DP','Category','Loc','ID'])
avr_rsp = ac_infos['psth'][:,:,160:320].sum(-1)
avr_rsp = avr_rsp/avr_rsp.max(1,keepdims = True)

stim_seq = Load_Info('Metamer_Singlebubble_v251107')[0].loc[300:].reset_index(drop=True)

#%%
bubble_ratio = pd.DataFrame(0.0,columns = ['Cell','Graph','Ratio'],index = range(len(avr_rsp)))
for i in range(len(avr_rsp)):
    cc_resp = avr_rsp[i,:]
    for j in range(1,21): # cycle item
        raw_id = stim_seq[stim_seq.Object==j]
        bubble_id = np.array(raw_id[raw_id.Category=='Occluded'].index)
        graph_id = np.array(raw_id[raw_id.Category=='Raw_Raw_Ani'].index)
        raw_avr_rsp = cc_resp[graph_id]
        bubble_s = cc_resp[bubble_id]
        # cc_ratio = 


#%%

