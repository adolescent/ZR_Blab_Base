'''
一些时序上的分析，观察到了metamer打乱会延迟第二个响应峰，

继而提出假设，会不会利用不同时间段的反应相互解码会得到不同的结果?

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
binned_msb = msb_ani_only.reshape((len(msb_ani_only),5,100,450)).mean(1)
binned_msb = binned_msb.reshape((len(msb_ani_only),100,225,2)).mean(-1)

msb_raw_rsps = pd.DataFrame(index=range(100000000),columns=['Cell','Time','Img','Constrain','Response'])
counter=0
for i in tqdm(range(len(binned_msb))):
    for j in range(5):
        for k in range(20):
            for l in range(225):
                cc_rsp = binned_msb[i,j*20+k,l]
                msb_raw_rsps.loc[counter,:] = [i,l*2-100,k,j,cc_rsp]
                counter+=1

msb_raw_rsps = msb_raw_rsps.dropna(how='any')
len(msb_raw_rsps)
msb_raw_rsps.Response = msb_raw_rsps.Response.astype('f8')



#%%
sns.lineplot(data=msb_raw_rsps,x='Time',hue='Constrain',y='Response',palette='tab10')


#%% get peak loc and info of different constrains.
from scipy.signal import find_peaks

df = msb_raw_rsps
results = []

# 设置绘图布局
fig, ax = plt.subplots(figsize=(10, 6))

# 按 Constrain 分组循环处理
for constr, group_df in df.groupby('Constrain'):
    
    # --- 步骤 A: 聚合 ---
    # 计算该 Constrain 下所有 Cell 在每个时刻的平均 Response
    # sort_values 确保时间顺序正确
    agg_df = group_df.groupby('Time')['Response'].mean().reset_index().sort_values('Time')
    
    x = agg_df['Time'].values
    y = agg_df['Response'].values
    
    # --- 步骤 B: 平滑 (关键步骤) ---
    # 使用 Savitzky-Golay 滤波器 (window_length 必须是奇数, polyorder 是多项式阶数)
    # 根据你的数据采样率调整 window_length，越长越平滑
    y_smooth = y
    
    # --- 步骤 C: 寻峰 ---
    # height: 最小峰值高度 (根据数据范围调整)
    # distance: 两个峰之间的最小水平距离 (防止把一个宽峰识别为两个)
    # prominence: 突起程度，过滤掉背景噪音上的小波动
    peaks_indices, properties = find_peaks(y_smooth, height=0, distance=3, prominence=0.0005)
    
    # 获取峰值对应的时间
    peak_times = x[peaks_indices]
    peak_responses = y_smooth[peaks_indices]
    
    # --- 步骤 D: 筛选前后两个峰 ---
    # 逻辑：如果找到多于2个峰，取最显著的两个；或者按时间排序取前两个
    # 这里我们假设一定有两个主要峰，并按时间排序
    if len(peak_times) >= 2:
        # 这里的策略是：先按峰的突起程度(prominence)取前2名，再按时间排序
        # 这样可以避免选中了一个很大的峰和一个很小的噪音，而漏掉真正的第二峰
        prominences = properties['prominences']
        top2_indices_local = np.argsort(prominences)[-2:] # 取最大的两个的索引
        top2_indices_final = np.sort(peaks_indices[top2_indices_local]) # 恢复时间顺序
        
        t1, t2 = x[top2_indices_final]
        r1, r2 = y_smooth[top2_indices_final]
    else:
        # 如果没找到两个峰，填充 NaN
        t1, t2 = (peak_times[0], np.nan) if len(peak_times) == 1 else (np.nan, np.nan)
        r1, r2 = (peak_responses[0], np.nan) if len(peak_times) == 1 else (np.nan, np.nan)

    # 存入结果
    results.append({
        'Constrain': constr,
        'Time_Peak1': t1,
        'Time_Peak2': t2,
        'Response_Peak1': r1,
        'Response_Peak2': r2
    })
    
    # 绘图验证
    p = ax.plot(x, y_smooth, label=f'C{constr}')
    color = p[0].get_color()
    # 标记找到的峰
    if not np.isnan(t1): ax.plot(t1, r1, 'x', color=color, markersize=10)
    if not np.isnan(t2): ax.plot(t2, r2, 'o', color=color, markersize=8, fillstyle='none')

# 输出结果表
results_df = pd.DataFrame(results)
print(results_df)

# 绘图设置
# ax.set_title("Response vs Time by Constrain (Detected Peaks)")
ax.set_xlabel("Time")
ax.set_ylabel("Mean Response")
ax.legend()
plt.show()

#%%######################### ASB #############
'''
对ASB做一样的操作，但ASB没有第一个峰
'''
binned_asb = asb_ani_only.reshape((len(asb_ani_only),5,100,450)).mean(1)
binned_asb = binned_asb.reshape((len(asb_ani_only),100,225,2)).mean(-1)

asb_raw_rsps = pd.DataFrame(index=range(100000000),columns=['Cell','Time','Img','Constrain','Response'])
counter=0
for i in tqdm(range(len(binned_asb))):
    for j in range(5):
        for k in range(20):
            for l in range(225):
                cc_rsp = binned_asb[i,j*20+k,l]
                asb_raw_rsps.loc[counter,:] = [i,l*2-100,k,j,cc_rsp]
                counter+=1

asb_raw_rsps = asb_raw_rsps.dropna(how='any')
len(asb_raw_rsps)
asb_raw_rsps.Response = asb_raw_rsps.Response.astype('f8')

#%% get peak loc and info of different constrains.
from scipy.signal import find_peaks

df = asb_raw_rsps
results = []

# 设置绘图布局
fig, ax = plt.subplots(figsize=(10, 6))

# 按 Constrain 分组循环处理
for constr, group_df in df.groupby('Constrain'):
    
    # --- 步骤 A: 聚合 ---
    # 计算该 Constrain 下所有 Cell 在每个时刻的平均 Response
    # sort_values 确保时间顺序正确
    agg_df = group_df.groupby('Time')['Response'].mean().reset_index().sort_values('Time')
    
    x = agg_df['Time'].values
    y = agg_df['Response'].values
    
    # --- 步骤 B: 平滑 (关键步骤) ---
    # 使用 Savitzky-Golay 滤波器 (window_length 必须是奇数, polyorder 是多项式阶数)
    # 根据你的数据采样率调整 window_length，越长越平滑
    y_smooth = y
    
    # --- 步骤 C: 寻峰 ---
    # height: 最小峰值高度 (根据数据范围调整)
    # distance: 两个峰之间的最小水平距离 (防止把一个宽峰识别为两个)
    # prominence: 突起程度，过滤掉背景噪音上的小波动
    peaks_indices, properties = find_peaks(y_smooth, height=0, distance=3, prominence=0.0005)
    
    # 获取峰值对应的时间
    peak_times = x[peaks_indices]
    peak_responses = y_smooth[peaks_indices]
    
    # --- 步骤 D: 筛选前后两个峰 ---
    # 逻辑：如果找到多于2个峰，取最显著的两个；或者按时间排序取前两个
    # 这里我们假设一定有两个主要峰，并按时间排序
    if len(peak_times) >= 2:
        # 这里的策略是：先按峰的突起程度(prominence)取前2名，再按时间排序
        # 这样可以避免选中了一个很大的峰和一个很小的噪音，而漏掉真正的第二峰
        prominences = properties['prominences']
        top2_indices_local = np.argsort(prominences)[-2:] # 取最大的两个的索引
        top2_indices_final = np.sort(peaks_indices[top2_indices_local]) # 恢复时间顺序
        
        t1, t2 = x[top2_indices_final]
        r1, r2 = y_smooth[top2_indices_final]
    else:
        # 如果没找到两个峰，填充 NaN
        t1, t2 = (peak_times[0], np.nan) if len(peak_times) == 1 else (np.nan, np.nan)
        r1, r2 = (peak_responses[0], np.nan) if len(peak_times) == 1 else (np.nan, np.nan)

    # 存入结果
    results.append({
        'Constrain': constr,
        'Time_Peak1': t1,
        'Time_Peak2': t2,
        'Response_Peak1': r1,
        'Response_Peak2': r2
    })
    
    # 绘图验证
    p = ax.plot(x, y_smooth, label=f'C{constr}')
    color = p[0].get_color()
    # 标记找到的峰
    if not np.isnan(t1): ax.plot(t1, r1, 'x', color=color, markersize=10)
    if not np.isnan(t2): ax.plot(t2, r2, 'o', color=color, markersize=8, fillstyle='none')

# 输出结果表
results_df = pd.DataFrame(results)
print(results_df)

# 绘图设置
# ax.set_title("Response vs Time by Constrain (Detected Peaks)")
ax.set_xlabel("Time")
ax.set_ylabel("Mean Response")
ax.legend()
plt.show()

