'''

Analysis of stable stim of Depth cue Invariant.

'''

#%%

from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
import copy
import warnings
import gc
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB','.joblib')
save_path = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable'

site = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\DQInva','.joblib')[0]
a = JL.load(site)
#%% Load data, select good cells.
stim_info = a.stim_info
ani_cells,ani_psth = a.Cell_Selection(ceiling=0.2,prefer='Animate',dp_thres=0.5)
len(ani_cells)
redplot = ani_psth[:,:,150:500].sum(-1)
redplot_z = (redplot-redplot.mean(1,keepdims=True))/redplot.std(1,keepdims=True)
#%% FOB demo: show tuning pref of cells (24 Body | 24 Face | 24 Object)
fob_divs = [24, 48]

fig, ax = plt.subplots(figsize=(4, 6))
sns.heatmap(redplot_z[:, :72], center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False,
            cbar = False, ax=ax)
for x in fob_divs:
    ax.axvline(x, color='yellow', lw=2)
ax.set_xlabel('Body    |    Face    |  Object')
ax.set_ylabel('Neuron')
fig.tight_layout()


# sns.heatmap(redplot_z[:,72:216],center=0,vmax=3,vmin=-3)
#%%
y = ani_psth[:, 72:360, :901].mean((0, 1))  # -100~800 ms, 1 ms bins
t = np.arange(-100, -100 + y.size)
n = len(y) // 5 * 5
y5, t5 = y[:n].reshape(-1, 5).mean(1), t[:n].reshape(-1, 5).mean(1)
fig, ax = plt.subplots(figsize=(5, 4))
ax.axvspan(50, 400, color='lightgreen', alpha=0.25, zorder=0)
ax.plot(t5, y5 * 1e3, lw=2, color='C0')
ax.set_xlim(-100, 800)
ax.set_ylim(3.5, 7)
ax.axvline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Mean PSTH')
ax.text(0, 1.02, r'$\times 10^{-3}$', transform=ax.transAxes, ha='left', va='bottom')
fig.tight_layout()

#%% Select cell response in set 1, and average.
msk = (stim_info.Stim_Set == 'ShadingTex1') & (stim_info.Object != 0)
x = redplot[:, msk.to_numpy()]
set1_rsp = x.reshape(x.shape[0], 2,-1).mean(1)
set1_rsp_z = (set1_rsp-set1_rsp.mean(1,keepdims=True))/set1_rsp.std(1,keepdims=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(set1_rsp_z, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar_kws={'label': 'z-scored response'}, ax=ax)
ax.axvline(66, color='yellow', lw=2)
ax.set_xlabel('Stimulus index (Shading  |  Texture)')
ax.set_ylabel('Neuron')
fig.tight_layout()

np.save(ot.Join(save_path,'set1_rsp_z.npy'),set1_rsp_z)

#%% ############ Strength Compare ############
'''
shading - texture per image; cols 0:66 = shading, 66:132 = texture
It seems no significant difference between shading and texture.
'''
sh = stim_info.loc[msk & ~stim_info.Category.str.contains('_Tex', na=False)]
img_cols = (sh.Object.astype(int).astype(str) + '_' + sh.Category).values[:66]
shad_tex_diff = pd.DataFrame(
    set1_rsp_z[:, :66] - set1_rsp_z[:, 66:],
    index=ani_cells, columns=img_cols)

# paired t-test per image (neurons paired: shading vs texture)
from scipy import stats
t_stat, p_val = stats.ttest_rel(set1_rsp_z[:, :66], set1_rsp_z[:, 66:], axis=0)
tex_shad_ttest = pd.DataFrame({'t': t_stat, 'p': p_val, 'sig': p_val < 0.05}, index=img_cols)
sig_mat = tex_shad_ttest['sig'].to_numpy().reshape(1, -1)  # 1 x 66, True = significant


#%% ############ Strongest Response ############
figpath = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable\Stimset1'
import os
files = sorted(ot.Get_File_Name(figpath, '.png'))
m = set1_rsp_z.mean(0)
shad_ord, tex_ord = np.argsort(m[:66]), np.argsort(m[66:])

# 1. sorted mean response
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(m[:66][shad_ord][::-1], lw=2, label='Shading')
ax.plot(m[66:][tex_ord][::-1], lw=2, label='Texture')
ax.set_xlabel('Stimulus rank (strong → weak)')
ax.set_ylabel('Mean z-scored response')
ax.legend()
fig.tight_layout()

# 2. sorted image paths (shading / texture each 66)
sorted_shad = [files[i] for i in shad_ord]
sorted_tex = [files[i + 66] for i in tex_ord]
sorted_stim = pd.DataFrame({'rank': np.arange(1, 67), 'shading': sorted_shad, 'texture': sorted_tex})

# 3. top-10 montage
def show_top10(paths):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, fp in zip(axes.ravel(), paths[-10:][::-1]):
        ax.imshow(plt.imread(fp))
        ax.axis('off')
    fig.tight_layout()

show_top10(sorted_shad)
show_top10(sorted_tex)

#%%
'''Pearson r (across neurons): shading vs texture per Object × In/Out × Orientation'''
sub = stim_info.loc[msk]
sh = sub[~sub.Category.str.contains('_Tex', na=False)].drop_duplicates(['Object', 'Category'])[['Object', 'Category', 'FileName']]
tx = sub[sub.Category.str.contains('_Tex', na=False)].drop_duplicates(['Object', 'Category'])[['Object', 'Category', 'FileName']]
tx['Category'] = tx.Category.str.replace('_Tex', '', regex=False)
sh = sh.rename(columns={'FileName': 'FileName_sh'})
tx = tx.rename(columns={'FileName': 'FileName_tex'})
meta = sh.merge(tx, on=['Object', 'Category'])
fn2i = {os.path.basename(f): i for i, f in enumerate(files)}
parts = meta.Category.str.split('_', expand=True)
io, ori = parts[0], parts[1]
obj_shad_tex_corr = pd.DataFrame({
    'Object': meta.Object.astype(int).values,
    'Orientation': ori,
    'In/Out': io,
    'Corr': [np.corrcoef(set1_rsp_z[:, fn2i[a]], set1_rsp_z[:, fn2i[b]])[0, 1]
             for a, b in zip(meta.FileName_sh, meta.FileName_tex)],
})

#%% Shading only: In vs Out per obj (C-R-L); control = cross-obj at same C-R-L
sh66 = sh.copy()
p = sh66.Category.str.split('_', expand=True)
sh66['In/Out'], sh66['Orientation'] = p[0], p[1]
sh66['Object'] = sh66.Object.astype(int)
sh66['col'] = sh66.FileName_sh.map(fn2i)
v = lambda o, io, r: set1_rsp_z[:, sh66.query('Object==@o and `In/Out`==@io and Orientation==@r').col.iloc[0]]
rows = []
for obj in range(1, 12):
    for ori in 'C', 'R', 'L':
        others = [o for o in range(1, 12) if o != obj]
        corr_io = np.corrcoef(v(obj, 'In', ori), v(obj, 'Out', ori))[0, 1]
        ctrl = np.mean([np.corrcoef(v(obj, 'In', ori), v(o, 'In', ori))[0, 1] for o in others] +
                       [np.corrcoef(v(obj, 'Out', ori), v(o, 'Out', ori))[0, 1] for o in others])
        rows.append({'Object': obj, 'Orientation': ori, 'Corr_InOut': corr_io, 'Corr_ctrl': ctrl})
obj_inout_corr = pd.DataFrame(rows)

#%% bar plot: In/Out vs cross-obj control (paired)
t, p = stats.ttest_rel(obj_inout_corr.Corr_InOut, obj_inout_corr.Corr_ctrl)
fig, ax = plt.subplots(figsize=(4, 5))
x = [0, 1]
m = [obj_inout_corr.Corr_InOut.mean(), obj_inout_corr.Corr_ctrl.mean()]
e = [obj_inout_corr.Corr_InOut.sem(), obj_inout_corr.Corr_ctrl.sem()]
ax.bar(x, m, yerr=e, width=0.45, capsize=4, color=['steelblue', 'darkorange'])
for _, r in obj_inout_corr.iterrows():
    ax.plot(x, [r.Corr_InOut, r.Corr_ctrl], 'k-', lw=0.6, alpha=0.35)
ax.set_xticks(x, ['In vs Out', 'Cross-obj'])
ax.set_ylabel('Pearson r')
ax.set_title(f'paired t-test: p = {p:.4f}')
fig.tight_layout()

#%% Shading: C-R-L within obj vs cross-obj at same In/Out × orientation
rows = []
for obj in range(1, 12):
    for io in 'In', 'Out':
        vecs = [v(obj, io, r) for r in ['C', 'R', 'L']]
        corr_crl = np.mean([np.corrcoef(vecs[i], vecs[j])[0, 1] for i, j in ((0, 1), (0, 2), (1, 2))])
        others = [o for o in range(1, 12) if o != obj]
        ctrl = np.mean([np.corrcoef(vecs[k], v(o, io, r))[0, 1]
                        for o in others for k, r in enumerate('CRL')])
        rows.append({'Object': obj, 'In/Out': io, 'Corr_CRL': corr_crl, 'Corr_ctrl': ctrl})
obj_crl_corr = pd.DataFrame(rows)

#%% bar plot: C-R-L vs cross-obj control (paired)
t, p = stats.ttest_rel(obj_crl_corr.Corr_CRL, obj_crl_corr.Corr_ctrl)
fig, ax = plt.subplots(figsize=(4, 5))
x = [0, 1]
m = [obj_crl_corr.Corr_CRL.mean(), obj_crl_corr.Corr_ctrl.mean()]
e = [obj_crl_corr.Corr_CRL.sem(), obj_crl_corr.Corr_ctrl.sem()]
ax.bar(x, m, yerr=e, width=0.45, capsize=4, color=['steelblue', 'darkorange'])
for _, r in obj_crl_corr.iterrows():
    ax.plot(x, [r.Corr_CRL, r.Corr_ctrl], 'k-', lw=0.6, alpha=0.35)
ax.set_xticks(x, ['C-R-L', 'Cross-obj'])
ax.set_ylabel('Pearson r')
ax.set_title(f'paired t-test: p = {p:.4f}')
fig.tight_layout()

#%% ############ Part2,ShadingTex2 Analysis ############
set2_rsp = redplot[:,360:]
set2_rsp = set2_rsp.reshape(set2_rsp.shape[0], 2,-1).mean(1)
set2_rsp_z = (set2_rsp-set2_rsp.mean(1,keepdims=True))/set2_rsp.std(1,keepdims=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(set2_rsp_z, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar_kws={'label': 'z-scored response'}, ax=ax)
ax.set_ylabel('Neuron')
fig.tight_layout()

np.save(ot.Join(save_path,'set2_rsp_z.npy'),set2_rsp_z)
#%% re-arrange seq. 80×(body|face|fruit); per 80: 20obj×[Tex_CTR,Shading_CTR,Tex,Shading]
# new seq: shading-shading_ctr-texture-texture_ctr
import shutil
figpath2 = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable\Stimset2'
outpath = ot.Join(figpath2, 'sorted')
os.makedirs(outpath, exist_ok=True)
ord4 = [3, 1, 2, 0]  # → Shading, Shading_CTR, Tex, Tex_CTR
idx = [s + r * 4 + c for s in (0, 80, 160) for c in ord4 for r in range(20)]
info2 = stim_info.iloc[360:600].reset_index(drop=True)
sorted_stim2 = info2.iloc[idx].reset_index(drop=True)
sorted_stim2.insert(0, 'rank', np.arange(240))
for i, fn in enumerate(sorted_stim2.FileName):
    shutil.copy2(ot.Join(figpath2, fn), ot.Join(outpath, f'{i:03d}_{fn}'))
set2_rsp_z_sorted = set2_rsp_z[:, idx]


fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(set2_rsp_z_sorted, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar_kws={'label': 'z-scored response'}, ax=ax)
ax.axvline(80, color='yellow', lw=2)
ax.axvline(160, color='yellow', lw=2)
# ax.set_xlabel('Stimulus index (Shading  |  Texture)')
ax.set_ylabel('Neuron')
fig.tight_layout()

np.save(ot.Join(save_path,'set2_rsp_z_sorted.npy'),set2_rsp_z_sorted)

# %% Body / Face / Fruit: tex–shading + CTR pair (switchable)
ctr_mode = 'tex'  # 'shading': Shading vs Shading_CTR | 'tex': Tex vs Tex_CTR
# per 80 cols (sorted): 0–19 Shading, 20–39 Shading_CTR, 40–59 Tex, 60–79 Tex_CTR
ctr_off = {'shading': (0, 20, 'Shading vs Shading_CTR'), 'tex': (40, 60, 'Tex vs Tex_CTR')}
o1, o2, ctr_label = ctr_off[ctr_mode]
R = set2_rsp_z_sorted
rows = []
for sub, s in zip(['Body', 'Face', 'Fruit'], [0, 80, 160]):
    for i in range(20):
        sh, tx = s + i, s + 40 + i
        c1, c2 = s + o1 + i, s + o2 + i
        rows.append({
            'Subclass': sub,
            'Object': int(sorted_stim2.iloc[sh].Object),
            'Corr_tex_shad': np.corrcoef(R[:, sh], R[:, tx])[0, 1],
            'Corr_ctr': np.corrcoef(R[:, c1], R[:, c2])[0, 1],
        })
sub2_tex_shad_corr = pd.DataFrame(rows)

# bar plot: tex–shading vs CTR pair, by Body / Face / Fruit
fig, ax = plt.subplots(figsize=(5, 5))
subs, x0, w = ['Body', 'Face', 'Fruit'], np.arange(3) * 3, 0.35
ps = []
for i, sub in enumerate(subs):
    d = sub2_tex_shad_corr.query('Subclass == @sub')
    x = x0[i] + np.array([-w / 2, w / 2])
    ax.bar(x, [d.Corr_tex_shad.mean(), d.Corr_ctr.mean()], yerr=[d.Corr_tex_shad.sem(), d.Corr_ctr.sem()],
           width=w, capsize=4, color=['steelblue', 'darkorange'])
    for _, r in d.iterrows():
        ax.plot(x, [r.Corr_tex_shad, r.Corr_ctr], 'k-', lw=0.6, alpha=0.35)
    ps.append(stats.ttest_rel(d.Corr_tex_shad, d.Corr_ctr).pvalue)
ax.set_xticks(x0)
ax.set_xticklabels(subs)
for i, p in enumerate(ps):
    ax.text(x0[i], ax.get_ylim()[1] * 0.95, f'p={p:.3f}', ha='center', fontsize=9)
ax.set_ylabel('Pearson r')
ax.legend(['Tex vs Shading', ctr_label], loc='lower right')
fig.tight_layout()

#%%
'''Cross decode: train SVM on shading, test tex / shading_ctr / tex_ctr (per Body/Face/Fruit).'''
from sklearn.svm import SVC

R = set2_rsp_z_sorted
test_off = {'tex': 40, 'shading_ctr': 20, 'tex_ctr': 60}
rows = []
for sub, s in zip(['Body', 'Face', 'Fruit'], [0, 80, 160]):
    X_train = R[:, s + np.arange(20)].T
    y = np.arange(20)
    clf = SVC(kernel='linear',C=1).fit(X_train, y)
    for test_set, off in test_off.items():
        X_test = R[:, s + off + np.arange(20)].T
        acc = (clf.predict(X_test) == y).mean()
        rows.append({'Subclass': sub, 'Test_set': test_set, 'Accuracy': acc})
sub2_cross_decode = pd.DataFrame(rows)

sub2_cross_decode

#%%

