'''
This script will transform an example GoodUnit into an .npy file, making it 

'''

#%% load in good unit data
import mat73
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from Matrix_Tools import *
from scipy import stats
import OS_Tools as ot
import pandas as pd
from Spike_Tools import *

gn_path = r'D:\#Data\silct\GoodUnit_250411_JianJian_silct_npx_1416_g0_MSB.mat'
savepath= r'D:\#Data\silct\Trans_Response'
filename = gn_path.split('\\')[-1][:-4] # get whole name of current spike file.


# data_dict = mat73.loadmat(gn_path)
data_dict = ot.Load_Variable(savepath,filename+'.raw')
# ot.Save_Variable(savepath,filename,data_dict,'.raw')

# raw_data = data_dict['GoodUnitStrc']['response_matrix_img']
trail_info = data_dict['meta_data']['trial_valid_idx']  # all trails, remove 0 will return trains
raster_info = data_dict['GoodUnitStrc']['Raster']  # raster of all trails, average to get response matrix.
# calculate PSTH matrix.
cellnum = len(raster_info)
img_num = 1416 # given by 
time_points = (raster_info[0]).shape[1]
PSTH = np.zeros(shape = (cellnum,img_num,time_points),dtype='f8') # use average(not sum) for psth calculation.
for i in tqdm(range(cellnum)):
    cc_response = Spike_Arrange(raster_info[i],trail_info,img_num)
    PSTH[i,:,:] = cc_response.mean(0)
ot.Save_Variable(savepath,f'{filename}_PSTH',PSTH,'.psth')
#%% After calculation, comparing onset and base for response significance.
# use welch ttest for response strength calculation?
base = np.arange(75,125) # corresponding to -25-25ms
onset = np.arange(150,250) # corresponding to 50-250ms
diff_resp,_ = stats.ttest_ind(PSTH[:,:,onset],PSTH[:,:,base],axis=2)
diff_resp = np.nan_to_num(diff_resp)
diff_resp = Matrix_Clipper(diff_resp,percent=99.5)
v_bound = min(diff_resp.max(),abs(diff_resp.min()))
# plot parts here.
fig,ax = plt.subplots(ncols=1,nrows=2,sharex=True,dpi = 300)
sns.heatmap(diff_resp,center=0,vmax = v_bound,vmin = -v_bound,ax =ax[0],cbar=False)
# diff_resp[diff_resp<5]=0
ax[1].plot((diff_resp>2).sum(0))
# set ticks
ax[1].set_xticks([0,300,600,900,1200])
ax[1].set_xticklabels([0,300,600,900,1200])
#%% varify cell response, find light-tuned and response cell
locolizer_data = PSTH[:,1200:,:]
base = np.arange(75,125) # corresponding to -25-25ms
onset = np.arange(150,250) # corresponding to 50-150ms
p_thres = 0.01
prop_sig = 0.2
stable_r = 0.1
# ttest first
# _,light_tune = np.nan_to_num(stats.ttest_ind(locolizer_data[:,:,onset],locolizer_data[:,:,base],axis=2),nan=1)
_,light_tune = stats.ttest_ind(locolizer_data[:,:,onset].reshape(len(locolizer_data),-1),locolizer_data[:,:,base].reshape(len(locolizer_data),-1),axis=1)
print(f'Light Tuned Cell Num:{(light_tune<p_thres).sum()}')
light_tune_id = np.where(light_tune<p_thres)[0]
# plt.plot(light_tune)

## then test stability of half before and half after, if stable, use this cell.
half_before = np.zeros(shape = (cellnum,img_num,time_points),dtype='f8') 
half_after = np.zeros(shape = (cellnum,img_num,time_points),dtype='f8') 

for i in tqdm(range(cellnum)):
    cc_response = Spike_Arrange(raster_info[i],trail_info,img_num)
    half_before[i,:,:] = cc_response[:5].mean(0)
    half_after[i,:,:] = cc_response[5:].mean(0)
before_locolizer = half_before[:,1200:,:]
after_locolizer = half_after[:,1200:,:]
before_locolizer_resp,_ =stats.ttest_ind(before_locolizer[:,:,onset],before_locolizer[:,:,base],axis=2)
before_locolizer_resp =  np.nan_to_num(before_locolizer_resp,nan=0)
after_locolizer_resp,_ =stats.ttest_ind(after_locolizer[:,:,onset],after_locolizer[:,:,base],axis=2)
after_locolizer_resp =  np.nan_to_num(after_locolizer_resp,nan=0)
stables = np.zeros(cellnum)
r,p = stats.pearsonr(before_locolizer_resp,after_locolizer_resp,axis=1)
stable_id = np.where(r>stable_r)[0]

# join 2 ways, getting real tuned and stable cell ids.
tuned_cell = np.intersect1d(light_tune_id,stable_id)
tuned_response = diff_resp[tuned_cell,:]

fig,ax = plt.subplots(ncols=1,nrows=1,sharex=True,dpi = 300,figsize = (7,3))
sns.heatmap(tuned_response,center=0,vmax = v_bound,vmin = -v_bound,ax =ax,cbar=False)
# diff_resp[diff_resp<5]=0
ot.Save_Variable(savepath,f'{filename}_filted_cells',tuned_response,'.cell')
#%% Re-Arrange data, getting the category-ordered response.
tsv_path = r'D:\#Data\#stimuli\silct\silct_info.tsv'
# 读取TSV文件
stim_info = pd.read_csv(tsv_path, sep='\t')
custom_order = ['Boulder','Texture','Filled','Face','Object','Body']
stim_info['FOB'] = pd.Categorical(
    stim_info['FOB'], 
    categories=custom_order,  # 指定自定义顺序
    ordered=True              # 标记为有序分类
)
sorted_stim_info = stim_info.iloc[:,:].sort_values(by=['FOB','Index'],ascending=[True,True])
sorted_index = np.array(sorted_stim_info.index)
# reorder response by index.
reordered_response = tuned_response[:, sorted_index]

fig,ax = plt.subplots(ncols=1,nrows=1,sharex=True,dpi = 300,figsize = (7,3))
sns.heatmap(reordered_response,center=0,vmax = v_bound,vmin = -v_bound,ax =ax,cbar=False)

#%% scatter each cell's response, getting it's 













