'''
This script is used for event response matrix calculator.

'''
#%%
from mua_detection import MUA_Detector
import pandas as pd
from toolkits import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

wp=r'D:\#Data\Loc_Example\test_wp'
npx_path = r'D:\#Data\Loc_Example\npx_root_decoy\NPX_MD241029_exp_g0'
thres=1.5
base_time = np.arange(75,125) # -25~25ms
onset_time = np.arange(150,250) # 50~150ms


#%% load response time and 
tsv_path = Get_File_Name(wp,'.tsv')[0]
event_path = Get_File_Name(wp,'.csv','Onset_Times')[0]

stim_info = pd.read_csv(tsv_path, sep='\t')
event_time = pd.read_csv(event_path, sep=',')
event_time = event_time[event_time['Trail_ID'] != 0].reset_index(drop=True)



#%% get response matrix and transfer it into psth format.
response_matrix = MUA_Detector(wp,npx_path,thres)
#%%
before_time = 100
after_time = 350
channel_num = response_matrix.shape[0]
img_num = len(stim_info)
time_points = after_time+before_time
N_repeat = int(len(event_time)/img_num)
psth = np.zeros(shape = (channel_num,img_num,time_points),dtype='f8')
counter=np.zeros(img_num)
# cycle event time for 
for i in range(len(event_time)):
    c_event = event_time.iloc[i,:]
    c_times = np.arange((c_event['Onset_Time']-before_time),(c_event['Onset_Time']+after_time))
    c_id = c_event['Trail_ID']-1
    if counter[c_id] == N_repeat:
        continue
    else:
        psth[:,c_id,:] += response_matrix[:,c_times]
        counter[c_id] += 1

#%% calculate all d prime.
d_primes = np.zeros(shape = (channel_num,img_num))
for i in range(channel_num):
    for j in range(img_num):
        c_off = psth[i,j,base_time]
        c_on = psth[i,j,onset_time]
        d_primes[i,j]= D_Prime(c_off,c_on)
d_primes = np.nan_to_num(d_primes)
sns.heatmap(d_primes,cmap='bwr')

#%% test subtraction between classes
clusts = list(set(stim_info['FOB']))
print(f'All Clusts:{clusts}')
A_sets = [3]
B_sets = [1,2,4,5,6]
A_sets_real = [clusts[i-1] for i in A_sets]
B_sets_real = [clusts[i-1] for i in B_sets]
print(f'A Set: {A_sets_real}')
print(f'B Set: {B_sets_real}')

face_ids = np.array(stim_info[stim_info['FOB'].isin(A_sets_real)].index)
noface_ids = np.array(stim_info[stim_info['FOB'].isin(B_sets_real)].index)

#%%
channel_face_resp = np.zeros(len(d_primes))
for i in range(channel_num):
    channel_face_resp[i]=D_Prime(d_primes[i,noface_ids],d_primes[i,face_ids])
channel_face_resp = np.nan_to_num(channel_face_resp)
plt.scatter(x=channel_face_resp,y=range(len(d_primes)),s=10)
plt.axvline(x=d_primes.mean(),linestyle='--',c='gray')
plt.xlabel(f'd prime of {A_sets_real}')