'''
This will try to show ASB 
'''

#%%
import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np
import copy


ceiled_files = ot.Get_File_Name(r'D:\#Data\Metamer\ceiled_response\ASB','.pkl')


#%%
for i,c_file in enumerate(ceiled_files):
    c_psth,c_info = ot.Load_Variable(c_file)
    c_body = c_info[c_info['Contra']=='Body_Contra'].reset_index(drop=True)
    c_body['Loc']=i
    if i == 0:
        concated_psth = copy.deepcopy(c_psth[:,72:,:])
        concated_info = copy.deepcopy(c_body)

#%% re-arrange cells with d prime.
body_dp = np.array(concated_info['D_Prime'])

body_dp_sorted = np.sort(body_dp)
sorted_indices = np.argsort(body_dp)
sorted_psth = concated_psth[sorted_indices]
# #%% calculate response of each cell.
# normed_psth = np.zeros(shape=sorted_psth.shape)
# for i in range(sorted_psth.shape[0]):
#     for j in range(sorted_psth.shape[1]):
#         normed_psth[i,j,:] = np.nan_to_num(sorted_psth[i,j,:]/sorted_psth[i,j,:].sum())
# redplot = normed_psth[:,:,150:200].sum(-1)
# sns.heatmap(redplot,center=0)

# calculate redplot of given series.
redplot = Redplot(sorted_psth)

#%% plot part
fig,ax = plt.subplots(nrows=1,ncols=2,figsize = (10,8),dpi=240,sharey=True)

sns.heatmap(redplot,center=0,ax = ax[1],vmax = 0.7,vmin = -0.7,cbar=False)
ax[0].barh(np.arange(len(body_dp_sorted)),body_dp_sorted)
ax[0].axvline(x=0.5,linestyle ='--',c='gray')
ax[1].set_xticks(np.arange(0,1040,40))
ax[1].set_xticklabels(np.arange(0,1040,40))

#%% save msb response.
ot.Save_Variable(r'D:\#Data\Metamer\ceiled_response\ASB','Sorted_Response',(body_dp_sorted,redplot,sorted_psth))

