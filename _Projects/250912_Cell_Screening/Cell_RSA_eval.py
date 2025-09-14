'''
This script will generate RSA of 200 different conditions, for each cell in MSB or ASB.
'''


#%%

import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np
import copy

msb_dp,msb_redplot,_ = ot.Load_Variable(r'D:\_DataTemp\Metamer\ceiled_response\ASB','Sorted_Response.pkl')

used_response = msb_redplot[np.where(msb_dp>0.5)[0][0]:,:]

#%%
avr_msb_resp = used_response.reshape(len(used_response),5,200).mean(1)


# then for each cell, calculate it's correlation matrix.
new_ids = []
for i in range(40):
    new_ids.extend([i,i+40,i+80,i+120,i+160])

# avr_msb_resp_arrange = avr_msb_resp[:,new_ids]

# generate all-cell corr matrix
global_corr_matrix = Corr_Matrix(avr_msb_resp)

fig,ax = plt.subplots(ncols=1,nrows=1,dpi=240,figsize=(6,6))
sns.heatmap(global_corr_matrix,center=0,cbar=False,square=True,xticklabels=False,yticklabels=False,ax=ax,vmax=0.8)
ax.set_yticks([0,40,80,120,160,200])
ax.set_yticklabels([0,40,80,120,160,200])
ax.set_xticks([0,40,80,120,160,200])
ax.set_xticklabels([0,40,80,120,160,200])

#%% caluclate alex response.
alex_resp = np.load(r'D:\_Codes\ZR_Blab_Base\_Projects\250821_Network_rsa\Metamer_Alexnet_fc6_resps.npy').T
avr_alex_resp = alex_resp.reshape(len(alex_resp),5,200).mean(1)


# then for each cell, calculate it's correlation matrix.
new_ids = []
for i in range(40):
    new_ids.extend([i,i+40,i+80,i+120,i+160])

# avr_msb_resp_arrange = avr_msb_resp[:,new_ids]

# generate all-cell corr matrix
global_corr_matrix = Corr_Matrix(avr_alex_resp)

fig,ax = plt.subplots(ncols=1,nrows=1,dpi=240,figsize=(6,6))
sns.heatmap(global_corr_matrix,center=0,cbar=False,square=True,xticklabels=False,yticklabels=False,ax=ax,vmax=0.8)
ax.set_yticks([0,40,80,120,160,200])
ax.set_yticklabels([0,40,80,120,160,200])
ax.set_xticks([0,40,80,120,160,200])
ax.set_xticklabels([0,40,80,120,160,200])


#%% calculate single-cell response.
all_asb_resp_matrix = np.zeros(shape = (len(used_response),5,5))
for i in range(len(used_response)):
    c_resp = used_response[i,:]
    # pivot response into N_fig*N_process.
    avr_resp = c_resp.reshape(5,200).mean(0)
    pivot_resp = avr_resp.reshape(5,40)
    cc_corr_matrix = Corr_Matrix(pivot_resp[:,:20].T)
    all_asb_resp_matrix[i,:,:] = cc_corr_matrix

fig,ax = plt.subplots(ncols=1,nrows=1,dpi=240,figsize=(6,6))
sns.heatmap(cc_corr_matrix,center=0,square=True,xticklabels=False,yticklabels=False,ax=ax,vmax=0.6,cbar=False)
ax.set_yticks(np.array([1,2,3,4,5])-0.5)
ax.set_yticklabels(['Raw','C4','C3','C2','C1'])
ax.set_xticks(np.array([1,2,3,4,5])-0.5)
ax.set_xticklabels(['Raw','C4','C3','C2','C1'])
# ax.set_xticks([0,40,80,120,160,200])
# ax.set_xticklabels([0,40,80,120,160,200])