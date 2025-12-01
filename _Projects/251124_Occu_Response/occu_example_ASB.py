'''
This scrip will show the response area of occulution, indicating the most activated area of 

For quick report, here we use no class.
All data need further analysis later.

'''



#%%
from Spike_Tools import *
from Py_Structure.Info_Files.InfoLoader import Load_Info
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageEnhance



gn_path = r'E:\#Preprocessed_Data\GoodUnits\GoodUnit_251110_ZhuangZhuang_Metamer_Singlebubble_v251107_4540_g1_AL_ASB.mat'
stim_info,stim_masks,_ = Load_Info(r'Metamer_Singlebubble_v251107',load_mask=True)
# psth = PSTH_From_Goodunit(gn_path,img_num=4540)
# np.savez_compressed(r'E:\#Preprocessed_Data\GoodUnits\GoodUnit_251110_ZhuangZhuang_Metamer_Singlebubble_v251107_4540_g1_AL_ASB',psth = psth)
psth = np.load(r'E:\#Preprocessed_Data\GoodUnits\GoodUnit_251110_ZhuangZhuang_Metamer_Singlebubble_v251107_4540_g1_AL_ASB.npz')['psth']

#%% do noise ceiling for given stim, using all stim set, odd&end ceiling
# test ceiling results vs fob ceiling.
ceiling_thres = 0.25
all_ceiling = odd_end_ceiling(psth,used_time=np.arange(160,320))
ok_cells = np.where(all_ceiling>ceiling_thres)[0]

ceiled_response = psth[ok_cells,:,:,160:320].sum(-1).mean(1)
ceiled_rasters = psth[ok_cells,:,:,:].mean(1)

ceiled_response = ceiled_response[np.concatenate((np.arange(125,210),np.arange(350,430)))]# remove AL
ceiled_rasters = ceiled_rasters[np.concatenate((np.arange(125,210),np.arange(350,430))),:,:]

#%%
ceiled_response_z = np.zeros(shape = (len(ok_cells),4540))
for i in range(len(ok_cells)):
    c_responses = ceiled_response[i,:]
    ceiled_response_z[i,:] = np.clip((c_responses-c_responses.mean())/c_responses.std(),-10,10)
#%%
response_arrange,idy = Redplot_PCA_Arranger(ceiled_response_z,reverse=True)
#%% plot and annotate response of 
N_sub = 9
fig,ax = plt.subplots(ncols=1,nrows=1,dpi=240,figsize=(2,4))
# sns.heatmap(response_arrange[:,1340:2940],center=0,vmax=5,cbar=False,ax = ax)
# sns.heatmap(response_arrange[:,1340:],center=0,vmax=5,cbar=False,ax = ax,cmap='bwr')
sns.heatmap(response_arrange[:,1340+80*(N_sub-1):1340+80*(N_sub)],center=0,vmax=5,cbar=False,ax = ax,cmap='bwr')
# sns.heatmap(response_arrange[:,2940:],center=0,vmax=5,cbar=False,ax = ax)
# sns.heatmap(response_arrange[:,2940+80*(N_sub-1):2940+80*(N_sub)],center=0,vmax=5,cbar=False,ax = ax,cmap='bwr')


# ax.plot([1599,1599],[0,449],c='r',zorder=10,linewidth=1)
# ax.set_yticks([])
# ax.set_xticks(np.arange(0,3200,80)+40)
# ax.set_xticklabels(np.concatenate((np.arange(1,21),np.arange(1,21))),fontsize=6,rotation=0)

#%% 
'''
So, the most effective element is RF or feature? We test single cell's response of different stimparts.
'''
#####   calculate weighted response roi of given 
## Bubble Parts First.

bubble_resp_mask = np.zeros(shape = (len(response_arrange),20,400,400),dtype='f8')
for i in tqdm(range(len(response_arrange))): # cycle cell
    for j in range(20): # cycle stimset
        cc_mask = np.zeros(shape=(400,400),dtype='f8')
        cc_responses = response_arrange[i,1340+80*j:1340+80*(j+1)]
        cc_stimmasks = stim_masks[1340+80*j:1340+80*(j+1),:,:]
        for k in range(80):
            cc_mask += cc_stimmasks[k,:,:]*cc_responses[k]
            bubble_resp_mask[i,j,:,:]=cc_mask

#%% visualize rf and masked graph parts.
from PIL import Image
raw_fig_path = ot.Get_File_Name(r'E:\#Coding_traces\Stim_Generator_bubble_single_LC\raw_fig','.jpg')[:20]
raw_figs = []
for i,c_img in enumerate(raw_fig_path):
    raw_figs.append(Image.open(c_img))

savepath=r'E:\#Coding_traces\251125_Bubbles\ASB_Bubble_RF'
#%% save cells
thres = 2.5
for i in tqdm(range(len(response_arrange))): #cycle cell
    for j in range(20):
        img = raw_figs[j]
        c_resps = bubble_resp_mask[i,j,:,:]
        mask = c_resps>thres
        img_array = np.array(img)
        mask_array = np.array(mask)
        darkened = ImageEnhance.Brightness(img).enhance(0.1)
        darkened_array = np.array(darkened)
        result_array = np.where(mask_array[:, :, None], img_array, darkened_array)
        result = Image.fromarray(result_array.astype('uint8'))

        # plot parts
        fig,ax = plt.subplots(ncols=2,nrows=1,dpi=240,figsize=(7,4))
        sns.heatmap(c_resps,center=0,vmax=5,ax=ax[0],cbar=False,cmap='bwr',square=True,xticklabels=False,yticklabels=False)
        ax[1].imshow(result, vmin=0, vmax=255)
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        fig.tight_layout()
        fig.savefig(ot.Join(savepath,f'Cell{i+1}_Img{j+1}.png'))
        plt.close(fig)
#%% Rest parts, codes are almost the same.
'''
这里使用的方法能衡量的其实是刺激的“不重要性”，即抠掉一个mask之后，反应强度越强则越不重要。

'''

bubble_resp_mask = np.zeros(shape = (len(response_arrange),20,400,400),dtype='f8')
for i in tqdm(range(len(response_arrange))): # cycle cell
    for j in range(20): # cycle stimset
        cc_mask = np.zeros(shape=(400,400),dtype='f8')
        cc_responses = response_arrange[i,2940+80*j:2940+80*(j+1)]
        cc_stimmasks = stim_masks[2940+80*j:2940+80*(j+1),:,:]
        for k in range(80):
            cc_mask += (1-cc_stimmasks[k,:,:])*cc_responses[k] # occuluded part
            bubble_resp_mask[i,j,:,:]=cc_mask

savepath=r'E:\#Coding_traces\251125_Bubbles\ASB_Rest_RF'

#%% save cells
thres = 2.5
for i in tqdm(range(len(response_arrange))): #cycle cell
    for j in range(20):
        img = raw_figs[j]
        c_resps = bubble_resp_mask[i,j,:,:]
        mask = c_resps<-thres
        img_array = np.array(img)
        mask_array = np.array(mask)
        darkened = ImageEnhance.Brightness(img).enhance(0.1)
        darkened_array = np.array(darkened)
        result_array = np.where(mask_array[:, :, None], img_array, darkened_array)
        result = Image.fromarray(result_array.astype('uint8'))

        # plot parts
        fig,ax = plt.subplots(ncols=2,nrows=1,dpi=240,figsize=(7,4))
        sns.heatmap(c_resps,center=0,ax=ax[0],cbar=False,cmap='bwr',square=True,xticklabels=False,yticklabels=False)
        ax[1].imshow(result, vmin=0, vmax=255)
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        fig.tight_layout()
        fig.savefig(ot.Join(savepath,f'Cell{i+1}_Img{j+1}.png'))
        plt.close(fig)

#%% Time courses of stim response.
ac_response = psth[ok_cells,:,:,:].mean(1).mean(0)
#%%
fig,ax = plt.subplots(ncols=1,nrows=1,dpi=240,figsize=(6,8))

sns.heatmap(ac_response,center=0,vmax=0.015,ax=ax,cbar=False,cmap='bwr',xticklabels=False,yticklabels=False)
# sns.heatmap(ac_response[2940:,:],center=0,vmax=0.02,cbar=False,cmap='bwr',xticklabels=False,yticklabels=False)

ax.plot([100,100],[0,4540],c='r',zorder=10,linewidth=1)
ax.plot([400,400],[0,4550],c='b',zorder=10,linewidth=1)
ax.plot([350,350],[0,4550],c='gray',linestyle='--',zorder=10,linewidth=1)
# divide lines
for i,cloc in enumerate([300,1300,1340,2940]):
    ax.plot([0,450],[cloc,cloc],c='black',linestyle='-',zorder=10,linewidth=0.5)
# ax.plot([0,450],[1000,1000],c='black',linestyle='-',zorder=10,linewidth=0.5)
