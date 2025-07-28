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


data_dict = mat73.loadmat(gn_path)
# data_dict = ot.Load_Variable(savepath,filename+'.raw')
ot.Save_Variable(savepath,filename,data_dict,'.raw')

# raw_data = data_dict['GoodUnitStrc']['response_matrix_img']
trail_info = data_dict['meta_data']['trial_valid_idx']  # all trails, remove 0 will return trains
raster_info = data_dict['GoodUnitStrc']['Raster']  # raster of all trails, average to get response matrix.
# pivot raster map into 

# cellnum = len(raw_data)
# img_num,time_points = raw_data[0].shape

#%% transform raw into python-readable style
raw_graphs = np.zeros(shape = (cellnum,img_num,time_points),dtype='f8')
for i in tqdm(range(cellnum)):
    raw_graphs[i,:,:] = raw_data[i]

filename = gn_path.split('\\')[-1][:-4]
np.save(r'D:\#Data\silct\Trans_Response\\'+filename+'_RAW.npy',raw_graphs)
#%% calculate significance of each cell's response.
base = np.arange(75,125) # corresponding to -25-25ms
onset = np.arange(150,350) # corresponding to 50-250ms

# how to calculate diff? D prime or just subtract with base?
diff_resp,_ = stats.ttest_ind(raw_graphs[:,:,onset],raw_graphs[:,:,base],axis=2)
diff_resp = np.nan_to_num(diff_resp)
diff_resp = Matrix_Clipper(diff_resp,percent=99.5)
# diff_resp.shape
v_bound = min(diff_resp.max(),abs(diff_resp.min()))

# diff_resp = (raw_graphs[:,:,onset].mean(-1)-raw_graphs[:,:,base].mean(-1))
# sns.heatmap(diff_resp,center=0,vmin = -20,vmax = 20)
# raster = diff_resp>5
# plt.plot((diff_resp).sum(0))
fig,ax = plt.subplots(ncols=1,nrows=2,sharex=True,dpi = 300)
sns.heatmap(diff_resp,center=0,vmax = v_bound,vmin = -v_bound,ax =ax[0],cbar=False)
# diff_resp[diff_resp<5]=0
ax[1].plot((diff_resp>5).sum(0))
# np.save(savepath+'\\'+filename+'_Response',diff_resp)

#%% Compare stimulus set, getting right stimulus infos 
tsv_path = r'D:\#Data\#stimuli\silct\silct_info.tsv'
# 读取TSV文件
stim_info = pd.read_csv(tsv_path, sep='\t')
all_stim_types = list(set(stim_info['FOB']))
texture_id = np.array(stim_info.groupby('FOB').get_group('Texture').index)
# generate a melted response id, a dumby method, but it works.
lines = diff_resp.shape[0]*diff_resp.shape[1]
response_matrix = pd.DataFrame(index=range(lines),columns=['Cell','Stim_File','FOB','Response'])
counter=0
for i in tqdm(range(diff_resp.shape[0])): # cycle cell
    for j in range(diff_resp.shape[1]): # cycle graph
        c_response = diff_resp[i,j]
        c_info = stim_info.iloc[j,:]
        response_matrix.loc[counter]=[i,c_info['FileName'],c_info['FOB'],c_response]
        counter +=1
response_matrix['Response'] = response_matrix['Response'] .astype('f8')
response_matrix['Cell'] = response_matrix['Cell'] .astype('i4')
ot.Save_Variable(savepath,f'{filename}_Frame',response_matrix,'.pd')

#%% plot several ez graphs 
tex_response = response_matrix[response_matrix['FOB'].isin(['Texture'])]
fob_response = response_matrix[response_matrix['FOB'].isin(['Face','Body','Object'])]


