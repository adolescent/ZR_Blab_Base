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
from Matrix_Tools import *
import cv2

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

#%% scatter each cell's response, getting it's best response toward a category.
text_id = np.array(stim_info[stim_info['FOB'].isin(['Texture'])].index)
boulder_id = np.array(stim_info[stim_info['FOB'].isin(['Boulder'])].index)
filled_id = np.array(stim_info[stim_info['FOB'].isin(['Filled'])].index)
fob_id = np.array(stim_info[stim_info['FOB'].isin(['Face','Body','Object'])].index)
text_resp = tuned_response[:,text_id]
boulder_resp = tuned_response[:,boulder_id]
filled_resp = tuned_response[:,filled_id]
fob_resp = tuned_response[:,fob_id]
prop = 0.2
text_best = Get_Extreme(text_resp,'max',1,prop)
boulder_best = Get_Extreme(boulder_resp,'max',1,prop)
filled_best = Get_Extreme(filled_resp,'max',1,prop)
fob_best = Get_Extreme(fob_resp,'max',1,prop)
# getting a scattered response, and plot 
best_resp_frame = pd.DataFrame(0.0,index=range(len(text_best)),columns = ['Texture','Filled','Boulder','FOB'])
for i in range(len(text_resp)):
    best_resp_frame.iloc[i,:] = [text_best[i],filled_best[i],boulder_best[i],fob_best[i]]

fig,ax = plt.subplots(nrows=2,ncols=2,dpi=240,figsize = (7,7))
for i in range(2):
    for j in range(2):
        ax[i][j].plot([0,5],[0,5],linestyle='--',color='gray',alpha=0.7)
        ax[i][j].set_ylim(0,5)
        ax[i][j].set_xlim(0,5)

sns.scatterplot(data = best_resp_frame,x='FOB',y='Filled',ax = ax[0][0],lw=0,s=3)
sns.scatterplot(data = best_resp_frame,x='Texture',y='Filled',ax = ax[0][1],lw=0,s=3)
sns.scatterplot(data = best_resp_frame,x='Boulder',y='Texture',ax = ax[1][0],lw=0,s=3)
sns.scatterplot(data = best_resp_frame,x='Boulder',y='Filled',ax = ax[1][1],lw=0,s=3)

fig.tight_layout()

#%%######################### Another version, figure-based analysis
'''
Here we try to analyze graph's respone on 3 different situations.

'''
figure_response_frame = pd.DataFrame(index=range(text_resp.shape[1]),columns = ['Boulder_vec','Texture_vec','Filled_vec','Boulder_Resp','Filled_Resp','Texture_Resp','Corr_FB','Corr_FT','Corr_BT'])

# get response vector of each figure, and calculate similarity.
strength_prop = 0.5
k = int(len(tuned_response)*strength_prop)
for i in range(text_resp.shape[1]):
    c_text_vec = tuned_response[:,i*3]
    c_boulder_vec = tuned_response[:,i*3+1]
    c_filled_vec = tuned_response[:,i*3+2]
    c_text_resp = np.partition(c_text_vec, -k)[-k:].mean()
    c_boulder_resp = np.partition(c_boulder_vec, -k)[-k:].mean()
    c_filled_resp = np.partition(c_filled_vec, -k)[-k:].mean()
    # calculate pearsonr 
    c_bt,_ = stats.pearsonr(c_boulder_vec,c_text_vec)
    c_fb,_ = stats.pearsonr(c_filled_vec,c_boulder_vec)
    c_ft,_ = stats.pearsonr(c_filled_vec,c_text_vec)
    figure_response_frame.loc[i,:] = [c_boulder_vec,c_text_vec,c_filled_vec,c_boulder_resp,c_filled_resp,c_text_resp,c_fb,c_ft,c_bt]
    
figure_response_frame['Boulder_Resp'] = figure_response_frame['Boulder_Resp'].astype('f8')
figure_response_frame['Filled_Resp'] = figure_response_frame['Filled_Resp'].astype('f8')
figure_response_frame['Texture_Resp'] = figure_response_frame['Texture_Resp'].astype('f8')
figure_response_frame['Corr_FB'] = figure_response_frame['Corr_FB'].astype('f8')
figure_response_frame['Corr_FT'] = figure_response_frame['Corr_FT'].astype('f8')
figure_response_frame['Corr_BT'] = figure_response_frame['Corr_BT'].astype('f8')



fig,ax = plt.subplots(nrows=2,ncols=2,dpi=240,figsize = (7,7))
sns.scatterplot(data = figure_response_frame,x='Boulder_Resp',y='Filled_Resp',lw=0,s=3,ax=ax[0][0])
sns.scatterplot(data = figure_response_frame,x='Texture_Resp',y='Filled_Resp',lw=0,s=3,ax=ax[0][1])
sns.scatterplot(data = figure_response_frame,x='Boulder_Resp',y='Texture_Resp',lw=0,s=3,ax=ax[1][0])
for i in range(2):
    for j in range(2):
        if not (i==1)*(j==1): 
            ax[i][j].plot([0,3],[0,3],linestyle='--',color='gray',alpha=0.7)
            ax[i][j].set_ylim(0,3)
            ax[i][j].set_xlim(0,3)
ax[1][1].hist(figure_response_frame['Boulder_Resp']-figure_response_frame['Texture_Resp'],bins=25)
ax[1][1].axvline(0,linestyle = '--',alpha=0.5,color = 'gray')
ax[1][1].set_xlabel('Boulder-Texture')
fig.tight_layout()
#%% plot 3 condition response similarity.
figure_response_frame['Graph_ID'] = np.array(figure_response_frame.index)+1
melted_frame = figure_response_frame.melt(id_vars='Graph_ID',value_vars = ['Corr_FB','Corr_FT','Corr_BT'],value_name='Correlation',var_name='Pair')
sns.histplot(data = melted_frame,x='Correlation',hue = 'Pair',bins = 30,lw=0,common_norm=True,common_bins=True,stat='density',alpha=0.5)

#%% Demonstration of boulder best and texture best
figpath = r'D:\#Data\#stimuli\silct\silct_npx_1416'
figure_response_frame['Boulder-Texture'] = figure_response_frame['Boulder_Resp']-figure_response_frame['Texture_Resp']
a = figure_response_frame.sort_values(by=['Boulder-Texture'],ascending=False)
boulder_figs = np.array(a.iloc[:18].index)
texture_figs = np.array(a.iloc[-18:].index)

# plot boulder figures.
fig,ax = plt.subplots(ncols=3,nrows=6,figsize = (12,12),dpi=180)
for i in range(18):
    c_graph_id = texture_figs[i] # x3+1 is the first texture graph.
    c_texture = str(10000+c_graph_id*3+1)[1:]+'.jpg'
    c_boulder = str(10000+c_graph_id*3+2)[1:]+'.jpg'
    c_texture_graph = cv2.imread(ot.Join(figpath,c_texture),0)
    c_boulder_graph = cv2.imread(ot.Join(figpath,c_boulder),0)
    c_compare = np.concatenate([c_texture_graph,c_boulder_graph],axis=1)
    ax[i//3,i%3].imshow(c_compare,cmap='gray')
    ax[i//3,i%3].set_yticks([])
    ax[i//3,i%3].set_xticks([])
fig.tight_layout()

#%% Demonstration of Filled-Texture-Boulder most similar 
a = figure_response_frame.sort_values(by=['Corr_FB'],ascending=False)
best_FB_sim = np.array(a.iloc[:18].index)
worst_FB_sim = np.array(a.iloc[-18:].index)

fig,ax = plt.subplots(ncols=3,nrows=6,figsize = (12,12),dpi=180)
for i in range(18):
    c_graph_id = worst_FB_sim[i] # x3+1 is the first texture graph.
    c_texture = str(10000+c_graph_id*3+2)[1:]+'.jpg'
    c_boulder = str(10000+c_graph_id*3+3)[1:]+'.jpg'
    c_texture_graph = cv2.imread(ot.Join(figpath,c_texture),0)
    c_boulder_graph = cv2.imread(ot.Join(figpath,c_boulder),0)
    c_compare = np.concatenate([c_texture_graph,c_boulder_graph],axis=1)
    ax[i//3,i%3].imshow(c_compare,cmap='gray')
    ax[i//3,i%3].set_yticks([])
    ax[i//3,i%3].set_xticks([])
fig.tight_layout()

#%%############################# RSA analysis, getting similarity between graphs.
# re-arrange graphs, in order 

vectors = np.concatenate([text_resp,boulder_resp,filled_resp],axis=1)
sim_matrix = Corr_Matrix(vectors)


fig,ax = plt.subplots(nrows=1,ncols=1,dpi=330,figsize=(6,6))
sns.heatmap(sim_matrix,center=0,ax = ax,square=True,vmax = 0.5,vmin = -0.35,cbar=False)
ax.plot([0,1200],[400,400],color='y',lw=1,alpha=0.5)
ax.plot([0,1200],[800,800],color='y',lw=1,alpha=0.5)
ax.plot([400,400],[0,1200],color='y',lw=1,alpha=0.5)
ax.plot([800,800],[0,1200],color='y',lw=1,alpha=0.5)

ax.set_xticks([0,200,400,600,800,1000,1200])
ax.set_xticklabels([0,200,400,600,800,1000,1200])
ax.set_yticks([0,200,400,600,800,1000,1200])
ax.set_yticklabels([0,200,400,600,800,1000,1200])
#%% find universal correlated stim between filled and texture-boulder.
# tex_fill_matrix = sim_matrix[800:,:400]
boulder_fill_matrix = sim_matrix[800:,400:800]
thres = 0.35
# corr_counts = pd.DataFrame((tex_fill_matrix>0.35).sum(0))
corr_counts = pd.DataFrame((boulder_fill_matrix>0.35).sum(0))
tex_index = corr_counts.sort_values(by=[0],ascending=False)
best_tex = np.array(tex_index.iloc[:36].index)

fig,ax = plt.subplots(ncols=6,nrows=6,figsize = (12,12),dpi=180)
for i in range(36):
    c_graph_id = best_tex[i] # x3+1 is the first texture graph.
    c_texture = str(10000+c_graph_id*3+2)[1:]+'.jpg'
    c_texture_graph = cv2.imread(ot.Join(figpath,c_texture),0)
    ax[i//6,i%6].imshow(c_texture_graph,cmap='gray')
    ax[i//6,i%6].set_yticks([])
    ax[i//6,i%6].set_xticks([])

fig.tight_layout()


#%% ############################## PCA Analysis ##################
'''
Use all neurons as 
'''


