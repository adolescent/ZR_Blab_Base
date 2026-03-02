'''
Only Animate metamer analyzed.


'''


#%%
# from Cell_Class import Cell_Infos
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
from tqdm import tqdm

wp=r'E:\#Preprocessed_Data\Selected_Cells'
result_path = r'E:\#Coding_traces\251219_Metamer_Ani_Only_Anis'

msb_infos = np.load(ot.Join(wp,'MSB_Cells_Metamer_Only.npz'),allow_pickle=True)
asb_infos = np.load(ot.Join(wp,'ASB_Cells_Metamer_Only.npz'),allow_pickle=True)

## cut data, getting animated only,and save animate matrix.
msb_resps = msb_infos['psth']
n_msb = msb_resps.shape[0]
temp_data = msb_resps.reshape(n_msb,25,40, 450)
msb_ani_only = temp_data[:, :, :20, :].reshape(n_msb, -1, 450)
msb_ani_avr = msb_ani_only[:,:,160:320].sum(-1)

np.savez_compressed(ot.Join(wp,'MSB_Metamer_Ani_only.npz'),psth=msb_ani_only)


asb_resps = asb_infos['psth']
n_asb = asb_resps.shape[0]
temp_data = asb_resps.reshape(n_asb, 25, 40, 450)
asb_ani_only = temp_data[:, :, :20, :].reshape(n_asb, -1, 450)
asb_ani_avr = asb_ani_only[:,:,160:320].sum(-1)
np.savez_compressed(ot.Join(wp,'ASB_Metamer_Ani_only.npz'),psth=asb_ani_only)

## if you need cell ids
msb_dps = pd.DataFrame(msb_infos['d_primes'],columns=['Cell','D_Prime','Category','Loc','Cell_ID'])
msb_fob = pd.DataFrame(msb_infos['response'],columns=['Cell','Category','Response','Loc','Cell_ID'])
asb_dps = pd.DataFrame(asb_infos['d_primes'],columns=['Cell','D_Prime','Category','Loc','Cell_ID'])
asb_fob = pd.DataFrame(asb_infos['response'],columns=['Cell','Category','Response','Loc','Cell_ID'])


#%% 
'''
Brief seeing of avr spikes, including raw response and z scored response. return re-arranged cell id.
'''

msb_ani_z = (msb_ani_avr-msb_ani_avr.mean(1,keepdims=True))/msb_ani_avr.std(1,keepdims=True)
asb_ani_z = (asb_ani_avr-asb_ani_avr.mean(1,keepdims=True))/asb_ani_avr.std(1,keepdims=True)

msb_ani_z_arr,_ = Redplot_PCA_Arranger(msb_ani_z,reverse=True)
asb_ani_z_arr,_ = Redplot_PCA_Arranger(asb_ani_z,reverse=True)
msb_ani_avr_arr,msb_ids = Redplot_PCA_Arranger(msb_ani_avr,reverse=True)
asb_ani_avr_arr,asb_ids = Redplot_PCA_Arranger(asb_ani_avr,reverse=True)

# visualize z score
fig,ax = plt.subplots(ncols=2,nrows=1,dpi=300,figsize=(8,6))
sns.heatmap(asb_ani_z_arr.reshape((asb_ani_z_arr.shape[0],5,100)).mean(1),center=0,vmax=3,vmin=-3,ax=ax[0],cbar=False,xticklabels=False,yticklabels=False,cmap='bwr')
sns.heatmap(msb_ani_z_arr.reshape((msb_ani_z_arr.shape[0],5,100)).mean(1),center=0,vmax=3,vmin=-3,ax=ax[1],cbar=False,xticklabels=False,yticklabels=False,cmap='bwr')
fig.tight_layout()

# visualize raw
fig,ax = plt.subplots(ncols=2,nrows=1,dpi=300,figsize=(8,6))
sns.heatmap(asb_ani_avr_arr.reshape((asb_ani_avr_arr.shape[0],5,100)).mean(1),center=0,vmax=7,vmin=-3,ax=ax[0],cbar=False,xticklabels=False,yticklabels=False,cmap='bwr')
sns.heatmap(msb_ani_avr_arr.reshape((msb_ani_avr_arr.shape[0],5,100)).mean(1),center=0,vmax=7,vmin=-3,ax=ax[1],cbar=False,xticklabels=False,yticklabels=False,cmap='bwr')
fig.tight_layout()


#%% 
'''
Response Stats, calculate response of each cell to each graph. Return a data frame, including each cell's response to all graphs.
'''
all_cell_resps = pd.DataFrame(index=range(10000000),columns=['Area','Cell','Img','Constrain','Response'])
counter=0

for N in tqdm(range(5)):
    for i in range(20):
        for j in range(5):
            for cc in range(len(msb_ani_avr_arr)):
                cc_resp = msb_ani_avr_arr[cc,N*100+i+j*20]
                all_cell_resps.iloc[counter,:]=['MSB',cc,i,j,cc_resp]
                counter+=1

for N in tqdm(range(5)):
    for i in range(20):
        for j in range(5):
            for cc in range(len(asb_ani_avr_arr)):
                cc_resp = asb_ani_avr_arr[cc,N*100+i+j*20]
                all_cell_resps.iloc[counter,:]=['ASB',cc,i,j,cc_resp]
                counter+=1

all_cell_resps = all_cell_resps.dropna(how='all')
all_cell_resps.to_csv(ot.Join(result_path,'All_Cell_Response.csv'))


#%%
'''
Average response of different brain area, all, top 1 and top5. 
They seems different.
'''
msb_resps_melted = all_cell_resps.groupby('Area').get_group('MSB')
asb_resps_melted = all_cell_resps.groupby('Area').get_group('ASB')

# global
fig,ax = plt.subplots(ncols=1,nrows=1,dpi=300,figsize=(5,4))
sns.lineplot(data=all_cell_resps,x='Constrain',y='Response',hue='Area',ax=ax)
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(['Raw','C4','C3','C2','C1'])
ax.set_yticks([1,1.2,1.4])

#%%
'''
此处判断不同的图片组 在被metamer 打乱情况下的相应变化。
'''
all_cell_resps_avr = all_cell_resps.groupby(['Area', 'Cell', 'Img', 'Constrain'])['Response'].mean().reset_index()
all_cell_resps_avr_sorted = all_cell_resps_avr.sort_values(by='Response', ascending=False)

# 分组平均
group_keys = ['Area', 'Cell', 'Constrain']

# 使用agg函数，一次性生成好的图标
final_frame = all_cell_resps_avr_sorted.groupby(group_keys)['Response'].agg(
    Top5_Mean    = lambda x: x.nlargest(5).mean(),
    Max_Value    = 'max',
    Bottom5_Mean = lambda x: x.nsmallest(5).mean(),
    Min_Value    = 'min'
).reset_index()

# 查看结果
print(final_frame.head())
#%% Plot parts
# global
fig,ax = plt.subplots(ncols=2,nrows=2,dpi=300,figsize=(9,7))
sns.lineplot(data=final_frame,x='Constrain',y='Top5_Mean',hue='Area',ax=ax[0,0])
sns.lineplot(data=final_frame,x='Constrain',y='Bottom5_Mean',hue='Area',ax=ax[0,1])
sns.lineplot(data=final_frame,x='Constrain',y='Max_Value',hue='Area',ax=ax[1,0])
sns.lineplot(data=final_frame,x='Constrain',y='Min_Value',hue='Area',ax=ax[1,1])

for i in range(2):
    for j in range(2):
        ax[i,j].set_xticks([0,1,2,3,4])
        ax[i,j].set_xticklabels(['Raw','C4','C3','C2','C1'])
fig.tight_layout()

#%% 把神经元挑出来，判断神经元id与刺激变化trend的关系
avr_asb = final_frame[final_frame.Area =='ASB']
avr_msb = final_frame[final_frame.Area =='MSB']

n_cell = len(set(avr_msb.Cell))

# resp_mats = np.zeros(shape = (n_cell,3))
resp_mats_msb = pd.DataFrame(0.0,columns=['Cell','Slope','Intercept','R2','Body_dp','Face_dp','Object_dp'],index=range(n_cell))

for i in tqdm(range(n_cell)):
    cc_rsp = avr_msb[avr_msb.Cell==i]
    cc_body = msb_dps.loc[(msb_dps['Cell_ID']==i)&(msb_dps['Category']=='Body'), 'D_Prime'].iloc[0]
    cc_face = msb_dps.loc[(msb_dps['Cell_ID']==i)&(msb_dps['Category']=='Face'), 'D_Prime'].iloc[0]
    cc_obj = msb_dps.loc[(msb_dps['Cell_ID']==i)&(msb_dps['Category']=='Object'), 'D_Prime'].iloc[0]

    y = cc_rsp.Max_Value.values
    y=y/y.max()
    x = cc_rsp.Constrain.values
    # 1. 获取斜率和截距
    slope, intercept = np.polyfit(x, y, 1)
    # 2. 计算 R^2 (决定系数)
    correlation_matrix = np.corrcoef(x, y)
    r_squared = correlation_matrix[0, 1]**2
    # resp_mats[i,:] = [slope,intercept,r_squared]
    resp_mats_msb.iloc[i,:] = [i,slope,intercept,r_squared,cc_body,cc_face,cc_obj]

# 对ASB计算相同的mode
n_cell = len(set(avr_asb.Cell))
resp_mats_asb = pd.DataFrame(0.0,columns=['Cell','Slope','Intercept','R2','Body_dp','Face_dp','Object_dp'],index=range(n_cell))
for i in tqdm(range(n_cell)):
    cc_rsp = avr_asb[avr_asb.Cell==i]
    cc_body = asb_dps.loc[(asb_dps['Cell_ID']==i)&(asb_dps['Category']=='Body'), 'D_Prime'].iloc[0]
    cc_face = asb_dps.loc[(asb_dps['Cell_ID']==i)&(asb_dps['Category']=='Face'), 'D_Prime'].iloc[0]
    cc_obj = asb_dps.loc[(asb_dps['Cell_ID']==i)&(asb_dps['Category']=='Object'), 'D_Prime'].iloc[0]

    y = cc_rsp.Max_Value.values
    y=y/y.max()
    x = cc_rsp.Constrain.values
    # 1. 获取斜率和截距
    slope, intercept = np.polyfit(x, y, 1)
    # 2. 计算 R^2 (决定系数)
    correlation_matrix = np.corrcoef(x, y)
    r_squared = correlation_matrix[0, 1]**2
    # resp_mats[i,:] = [slope,intercept,r_squared]
    resp_mats_asb.iloc[i,:] = [i,slope,intercept,r_squared,cc_body,cc_face,cc_obj]

#%%
bad_cells_asb = resp_mats_asb[resp_mats_asb.R2<0.4].Cell.values
good_cells_asb = resp_mats_asb[resp_mats_asb.R2>0.4].Cell.values
a = avr_asb[avr_asb.Cell.isin(good_cells_asb)]
b = avr_asb[avr_asb.Cell.isin(bad_cells_asb)]
sns.lineplot(data = a,x='Constrain',y='Max_Value')
sns.lineplot(data = b,x='Constrain',y='Max_Value')
#%%
bad_cells_msb = resp_mats_msb[resp_mats_msb.R2<0.4].Cell.values
good_cells_msb = resp_mats_msb[resp_mats_msb.R2>0.4].Cell.values
a = avr_msb[avr_msb.Cell.isin(good_cells_msb)]
b = avr_msb[avr_msb.Cell.isin(bad_cells_msb)]
sns.lineplot(data = a,x='Constrain',y='Max_Value')
sns.lineplot(data = b,x='Constrain',y='Max_Value')

#%%
'''
Good Cell和Bad Cell, 在响应强度上和Body偏好程度上差异不明显

'''


def Sparseness(responses):
    # define sparseness from Rolls & Tovee (1995).
    responses = np.array(responses)
    n = len(responses)

    # 防止除以 0 的情况（如果所有响应都为 0）
    if np.sum(responses) == 0:
        return 0.0
    
    # 计算分子中的两部分
    mean_r = np.mean(responses)
    mean_sq_r = np.mean(responses**2)
    
    # 原始稀疏度
    raw_sparseness = 1 - (mean_r**2 / mean_sq_r)
    # 归一化到 0-1 之间
    s = raw_sparseness / (1 - 1/n)
    return s
#%%
# 得到对20张图片的原始反应
all_raw_resp = all_cell_resps_avr.groupby('Constrain').get_group(0).reset_index(drop=True)
n_asb = len(asb_ani_only)
n_msb = len(msb_ani_only)
asb_raw_resp = all_raw_resp.groupby('Area').get_group('ASB')
msb_raw_resp = all_raw_resp.groupby('Area').get_group('MSB')

info_frames = pd.DataFrame(index=range(1000000),columns=['Area','Cell','R2','Goodness','Body_dp','Sparseness','Best_Response','Avr_Response','Base'])


# cycle asb
counter=0
for i in tqdm(range(n_asb)):
    # response parts
    cc_raw = asb_raw_resp[asb_raw_resp.Cell==i].Response.values
    cc_sparseness = Sparseness(cc_raw)
    cc_max = cc_raw.max()
    cc_avr = cc_raw.mean()
    # cell fitting parts
    resp_mats_asb.Cell = resp_mats_asb.Cell.astype('i4')
    cc_fitts = resp_mats_asb[resp_mats_asb.Cell==i]
    cc_r2 = cc_fitts.R2.values[0]
    cc_body_dp = cc_fitts.Body_dp.values[0]
    if i in good_cells_asb:
        cc_good = 1
    else:
        cc_good = -1
    # get base line response of all animate graph.
    cc_base = asb_ani_only[i,:,75:125].mean()
    info_frames.loc[counter,:] = ['ASB',i,cc_r2,cc_good,cc_body_dp,cc_sparseness,cc_max,cc_avr,cc_base]
    counter +=1

# cycle msb
for i in tqdm(range(n_msb)):
    # response parts
    cc_raw = msb_raw_resp[msb_raw_resp.Cell==i].Response.values
    cc_sparseness = Sparseness(cc_raw)
    cc_max = cc_raw.max()
    cc_avr = cc_raw.mean()
    # cell fitting parts
    resp_mats_msb.Cell = resp_mats_msb.Cell.astype('i4')
    cc_fitts = resp_mats_msb[resp_mats_msb.Cell==i]
    cc_r2 = cc_fitts.R2.values[0]
    cc_body_dp = cc_fitts.Body_dp.values[0]
    if i in good_cells_msb:
        cc_good = 1
    else:
        cc_good = -1
    # get base line response of all animate graph.
    cc_base = msb_ani_only[i,:,75:125].mean()
    info_frames.loc[counter,:] = ['MSB',i,cc_r2,cc_good,cc_body_dp,cc_sparseness,cc_max,cc_avr,cc_base]
    counter +=1
info_frames = info_frames.dropna(how='any')
#%%
asb_resps = info_frames.groupby('Area').get_group('ASB')
msb_resps = info_frames.groupby('Area').get_group('MSB')

# sns.scatterplot(data = asb_resps,x='R2',y='Body_dp',hue='Goodness',s=3,lw=0,palette='tab10')
# sns.scatterplot(data = asb_resps,x='R2',y='Sparseness',hue='Goodness',s=3,lw=0,palette='tab10')
# sns.scatterplot(data = asb_resps,x='R2',y='Best_Response',hue='Goodness',s=3,lw=0,palette='tab10')
# sns.scatterplot(data = asb_resps,x='R2',y='Avr_Response',hue='Goodness',s=3,lw=0,palette='tab10')
# sns.scatterplot(data = asb_resps,x='R2',y='Base',hue='Goodness',s=3,lw=0,palette='tab10')
y = 'Best_Response'
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(3,4))
sns.set_theme(style="ticks")
# 1. 先画 Stripplot (底层)
sns.stripplot(
    data=info_frames, 
    y=y, x='Area', hue='Goodness', 
    palette='tab10', dodge=True, 
    s=2, linewidth=0, alpha=1,
    legend=False, # 避免图例重复
    ax=ax,zorder=1
)

# 2. 后画 Boxplot (顶层)
# 注意：要把 box_props 的 alpha 调低，或者将 facecolor 设为透明，否则会完全盖住点
sns.boxplot(
    data=info_frames, 
    y=y, x='Area', hue='Goodness', 
    palette='tab10', showfliers=False,
    boxprops={'alpha': 0.9}, # 设置 0.79 的透明度，让底层的点若隐若现
    ax=ax,zorder=2
)

plt.show()

## stats of diff from asb and msb, good and bad.
asb_a = asb_resps[asb_resps.Goodness==1][y].values.astype('f8')
asb_b = asb_resps[asb_resps.Goodness==-1][y].values.astype('f8')
print(stats.ttest_ind(asb_a,asb_b))

msb_a = msb_resps[msb_resps.Goodness==1][y].values.astype('f8')
msb_b = msb_resps[msb_resps.Goodness==-1][y].values.astype('f8')
print(stats.ttest_ind(msb_a,msb_b))

