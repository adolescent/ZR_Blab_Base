
#%%
import OS_Tools as ot
import numpy as np
from Common_Functions.Useful_Plotter import *




all_cell_path = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
check_area = 'AL'


area_dir = ot.Join(all_cell_path, check_area)

#%% ######## 原始FOB 响应 ########

fob = np.load(ot.Join(area_dir, 'fob_avr.npy'))
n_valid = np.sum(~np.isnan(fob), axis=1)
# FOB72: Body-Face-Object -> Face-Body-Object
i72 = n_valid == 72
f72 = fob[i72, :72][:, np.r_[24:48, 0:24, 48:72]]

# STI150: 两个连续 cycle（各75张）取平均，顺序 Face-Body-Object-Scene-Food
i150 = n_valid == 150
f150 = fob[i150, :150].reshape(-1, 2, 75).mean(1)

# 上 FOB72，下 STI150；右侧补 NaN 对齐 75 列
hm = np.vstack([np.c_[f72, np.full((f72.shape[0], 3), np.nan)], f150])
z = (hm - np.nanmean(hm, 1, keepdims=True)) / np.nanstd(hm, 1, keepdims=True)

fig, ax = plt.subplots(figsize=(4,6))
sns.heatmap(z, cmap='RdBu_r', center=0, vmin=-3, vmax=3,
            xticklabels=False, yticklabels=False, ax=ax)
n72, n_row = f72.shape[0], hm.shape[0]
for x in (24, 48):                          # FOB72 上区，每类 24 张
    ax.vlines(x, 0, n72, color='gold', lw=1.5)
for x in (15, 30, 45, 60):                  # STI150 下区，每 15 张一根
    ax.vlines(x, n72, n_row, color='gold', lw=1.5)
ax.axhline(n72, color='k', lw=2)
ax.set_xlabel('Face | Body | Object | Scene | Food ',size=10)
ax.set_ylabel(f'{check_area}  N Cell = {len(n_valid)}')
fig.tight_layout()
plt.show()

#%% ######## 平均后的Metamer打乱相应 ########
avr = np.load(ot.Join(area_dir, 'avr_rsp.npy'))
hm = avr.reshape(-1, 5, 200).mean(1)          # 5 cycle 平均 -> (N_cell, 200)
z = (hm - hm.mean(1, keepdims=True)) / hm.std(1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(z, cmap='RdBu_r', center=0, vmin=-3, vmax=3,
            xticklabels=False, yticklabels=False, ax=ax)
for x in (40, 80, 120, 160):                  # raw | s1 | s2 | s3 | s4，各 40 张
    ax.axvline(x, color='gold', lw=1.5)
ax.set_xlabel('Raw | Shuffle1 | Shuffle2 | Shuffle3 | Shuffle4')
ax.set_ylabel(f'{check_area}  N Cell = {hm.shape[0]}')
fig.tight_layout()
plt.show()
#%% 平均响应和raster map的统计，体现metamer的影响
img_id = [31]              # 1-based，40 张 raw 内的编号；可改为 [1, 3, 5] 取平均
bin_ms = 5
t_ms = np.arange(-100, -100 + 450)

psth = np.load(ot.Join(area_dir, 'psth.npy'))
ids = np.atleast_1d(img_id) - 1                       # 转 0-based
offs = [0, 40, 80, 120, 160]                          # raw, s1~s4
labels = ['Raw', 'S1', 'S2', 'S3', 'S4']

n_t = psth.shape[-1] // bin_ms * bin_ms
t_plot = t_ms[:n_t].reshape(-1, bin_ms).mean(1)

fig, ax = plt.subplots(figsize=(4, 3))
for off, lab in zip(offs, labels):
    cols = np.r_[[c * 200 + off + i for c in range(5) for i in ids]]
    pop = psth[:, cols, :n_t].mean(1)                 # 跨 neuron 前先对 cycle/img 平均
    fr = pop.reshape(pop.shape[0], -1, bin_ms).mean(-1) * 1000   # Hz
    m, sem = fr.mean(0), fr.std(0) / np.sqrt(fr.shape[0])
    line, = ax.plot(t_plot, m, lw=1.8, label=lab)
    ax.fill_between(t_plot, m - sem, m + sem, color=line.get_color(), alpha=0.2, linewidth=0)

ax.axvline(0, color='gray', ls='--', lw=0.8)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate (Hz)')
ax.set_title(f'{check_area}  img {img_id[0]} ')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
#%% ######## metamer 相对 raw 的响应比例（四脑区） ########
import pandas as pd

stim_part = 'inani'                             # 'ani': id 1-20, 'inani': id 21-40
img_ids = np.arange(0, 20) if stim_part == 'ani' else np.arange(20, 40)
base_ids = np.arange(0, 20)                     # 固定用 ani 的 raw 平均做 normalization
shuf_labels = ['Raw', 'S1', 'S2', 'S3', 'S4']
colors = {'ML': '#2166ac', 'MSB': '#67a9cf', 'AL': '#b2182b', 'ASB': '#ef8a62'}  # M蓝 / A红
x = np.arange(len(shuf_labels))

fig, ax = plt.subplots(figsize=(5, 4))
all_ratio = []
for area in ['ML', 'MSB', 'AL', 'ASB']:
    r = np.load(ot.Join(ot.Join(all_cell_path, area), 'avr_rsp.npy')).reshape(-1, 5, 5, 40).mean(1)
    cell_mean = r[:, :, img_ids].mean(-1)
    raw_ani = r[:, 0, base_ids].mean(-1, keepdims=True)
    ratio = cell_mean / raw_ani
    ratio[raw_ani[:, 0] <= 0] = np.nan
    df = pd.DataFrame(ratio, columns=shuf_labels)
    df['Cell'], df['Area'] = np.arange(len(ratio)), area
    all_ratio.append(df.melt(id_vars=['Cell', 'Area'], var_name='Shuffle', value_name='Ratio'))

    m = df[shuf_labels].mean()
    sem = df[shuf_labels].sem()
    ax.errorbar(x, m, yerr=sem, fmt='o-', lw=1.8, capsize=3, color=colors[area], label=area)

df_ratio = pd.concat(all_ratio, ignore_index=True)
ax.axhline(1, color='gray', ls='--', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(shuf_labels)
ax.set_xlabel('Shuffle level')
ax.set_ylabel('Response / Ani Raw')
ax.set_title(f'{stim_part} (norm by ani raw)')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

#%%



