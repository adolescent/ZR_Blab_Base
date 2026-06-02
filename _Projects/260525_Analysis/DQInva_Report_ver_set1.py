'''
Based on initial results, let's summarize the main findings and conclusions.
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
set1_stims = np.load(ot.Join(save_path,'set1_rsp_z.npy'),allow_pickle=True)

#%%
# orig col/files: per obj × [In-C,In-R,In-L, Out-C,Out-R,Out-L] (CRL-In then CRL-Out); 0:66 shading, 66:132 texture
# new:  per block × 11 objs [In-R, Out-R, In-C, Out-C, In-L, Out-L] × (shading | texture)
_n_obj = 11
_cond_in_obj = (1, 4, 0, 3, 2, 5)  # In-R, Out-R, In-C, Out-C, In-L, Out-L  (R-C-L)
_reindex = np.array([o * 6 + c for c in _cond_in_obj for o in range(_n_obj)])
redplot_reindex = set1_stims[:, np.r_[_reindex, _reindex + 66]]

#%% heatmap (reindexed): In|Out yellow, C|R|L blue, shading|texture black
_cond_lbl = ['In-R', 'Out-R', 'In-C', 'Out-C', 'In-L', 'Out-L']

fig, ax = plt.subplots(figsize=(10, 5.5))
sns.heatmap(redplot_reindex, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar_kws={'shrink': 0.6}, ax=ax)
for x in (11, 33, 55, 77, 99, 121):
    ax.axvline(x, color='yellow', lw=1.5)
for x in (22, 44, 88, 110):
    ax.axvline(x, color='royalblue', lw=1.5, ls='-')
ax.axvline(66, color='k', lw=2)
for i, lab in enumerate(_cond_lbl * 2):
    ax.text(5.5 + i * _n_obj, 1.01, lab, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8)
ax.text(33, 1.07, 'Shading', transform=ax.get_xaxis_transform(), ha='center', fontsize=9)
ax.text(99, 1.07, 'Texture', transform=ax.get_xaxis_transform(), ha='center', fontsize=9)
ax.set_xlabel('Object (1–11 per block)')
ax.set_ylabel('Neuron')
fig.subplots_adjust(top=0.88)

#%%
# cross-correlation across 12 subclasses (each over 11 objects)
# subclass order follows redplot_reindex blocks:
# [In-R, Out-R, In-C, Out-C, In-L, Out-L] × [Shading, Texture]
_ori_map = {'R': 'R', 'C': 'C', 'L': 'L'}
_io_map = {'In': 'In', 'Out': 'Out'}
_st_map = {0: 'Shading', 1: 'Texture'}

subclass_rows = []
for st_i, st_name in _st_map.items():
    for i, lbl in enumerate(_cond_lbl):
        io_name, ori_name = lbl.split('-')
        subclass_rows.append({
            'subclass_id': st_i * len(_cond_lbl) + i,
            'subclass': f'{st_name}_{lbl}',
            'Shading/Texture': st_name,
            'In/Out': _io_map[io_name],
            'R-C-L': _ori_map[ori_name],
        })
subclass_meta = pd.DataFrame(subclass_rows)

# average across neurons, keep object profile (11 objs) for each subclass
mean_rsp = redplot_reindex.mean(axis=0).reshape(12, _n_obj)  # 12 subclasses × 11 objs

# pairwise correlation between subclasses (corr across the 11-object profile)
corr_mat = np.corrcoef(mean_rsp)  # 12 × 12

rows = []
for i in range(corr_mat.shape[0]):
    for j in range(i + 1, corr_mat.shape[1]):
        a = subclass_meta.iloc[i]
        b = subclass_meta.iloc[j]
        rows.append({
            'subclass_a': a['subclass'],
            'subclass_b': b['subclass'],
            'corr': corr_mat[i, j],
            'a_Shading/Texture': a['Shading/Texture'],
            'a_In/Out': a['In/Out'],
            'a_R-C-L': a['R-C-L'],
            'b_Shading/Texture': b['Shading/Texture'],
            'b_In/Out': b['In/Out'],
            'b_R-C-L': b['R-C-L'],
        })

# final result:
# - subclass_meta: each subclass's feature category
# - corr_df: pairwise cross-correlation between subclasses
corr_df = pd.DataFrame(rows).sort_values('corr', ascending=False).reset_index(drop=True)

#%% bar plots for three correlation analyses + shuffled-object controls
rng = np.random.default_rng(20260601)

cond_idx = {lab: i for i, lab in enumerate(_cond_lbl)}
ori_list = ['R', 'C', 'L']
io_list = ['In', 'Out']
st_list = ['Shading', 'Texture']
st_idx = {k: i for i, k in enumerate(st_list)}

# redplot_reindex columns are ordered as:
# (Shading 6 blocks x 11 obj) + (Texture 6 blocks x 11 obj)
X = redplot_reindex.reshape(redplot_reindex.shape[0], 2, 6, _n_obj)  # cell x st x cond x obj

def _safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

rows = []

# 1) matched Shading vs Texture (66 matched conditions = 11 obj x 6 cond)
for cond_lab in _cond_lbl:
    c = cond_idx[cond_lab]
    obj_perm = rng.permutation(_n_obj)
    for obj in range(_n_obj):
        real = _safe_corr(X[:, st_idx['Shading'], c, obj], X[:, st_idx['Texture'], c, obj])
        ctrl = _safe_corr(X[:, st_idx['Shading'], c, obj], X[:, st_idx['Texture'], c, obj_perm[obj]])
        rows.append({
            'analysis': 'Shading_vs_Texture',
            'unit': f'{cond_lab}_obj{obj+1}',
            'corr_real': real,
            'corr_ctrl': ctrl,
            'Shading/Texture': 'Across',
            'In/Out': cond_lab.split('-')[0],
            'R-C-L': cond_lab.split('-')[1],
            'Object': obj + 1,
        })

# 2) In vs Out within same shading/texture and same orientation (R/C/L)
for st in st_list:
    s = st_idx[st]
    for ori in ori_list:
        c_in = cond_idx[f'In-{ori}']
        c_out = cond_idx[f'Out-{ori}']
        obj_perm = rng.permutation(_n_obj)
        for obj in range(_n_obj):
            real = _safe_corr(X[:, s, c_in, obj], X[:, s, c_out, obj])
            ctrl = _safe_corr(X[:, s, c_in, obj], X[:, s, c_out, obj_perm[obj]])
            rows.append({
                'analysis': 'In_vs_Out',
                'unit': f'{st}_{ori}_obj{obj+1}',
                'corr_real': real,
                'corr_ctrl': ctrl,
                'Shading/Texture': st,
                'In/Out': 'Across',
                'R-C-L': ori,
                'Object': obj + 1,
            })

# 3) C-R-L pairwise corr within same shading/texture and same In/Out
ori_pairs = [('R', 'C'), ('R', 'L'), ('C', 'L')]
for st in st_list:
    s = st_idx[st]
    for io in io_list:
        for ori_a, ori_b in ori_pairs:
            c_a = cond_idx[f'{io}-{ori_a}']
            c_b = cond_idx[f'{io}-{ori_b}']
            obj_perm = rng.permutation(_n_obj)
            for obj in range(_n_obj):
                real = _safe_corr(X[:, s, c_a, obj], X[:, s, c_b, obj])
                ctrl = _safe_corr(X[:, s, c_a, obj], X[:, s, c_b, obj_perm[obj]])
                rows.append({
                    'analysis': 'CRL_pairwise',
                    'unit': f'{st}_{io}_{ori_a}-{ori_b}_obj{obj+1}',
                    'corr_real': real,
                    'corr_ctrl': ctrl,
                    'Shading/Texture': st,
                    'In/Out': io,
                    'R-C-L': f'{ori_a}-{ori_b}',
                    'Object': obj + 1,
                })

corr_detail_df = pd.DataFrame(rows)

summary_rows = []
for ana, d in corr_detail_df.groupby('analysis'):
    r = d['corr_real'].dropna()
    c = d['corr_ctrl'].dropna()
    summary_rows.extend([
        {'analysis': ana, 'group': 'Real', 'mean': r.mean(), 'sem': r.sem(), 'n': len(r)},
        {'analysis': ana, 'group': 'Control', 'mean': c.mean(), 'sem': c.sem(), 'n': len(c)},
    ])
corr_bar_df = pd.DataFrame(summary_rows)

# plot: single subplot (3 analyses together), clean style
order = ['Shading_vs_Texture', 'In_vs_Out', 'CRL_pairwise']
name_map = {
    'Shading_vs_Texture': 'Shading vs Texture',
    'In_vs_Out': 'In vs Out',
    'CRL_pairwise': 'C-R-L Pairwise',
}
plot_df = corr_detail_df.melt(
    id_vars=['analysis'], value_vars=['corr_real', 'corr_ctrl'],
    var_name='group', value_name='corr'
)
plot_df['group'] = plot_df['group'].map({'corr_real': 'Real', 'corr_ctrl': 'Control'})
plot_df['analysis_name'] = plot_df['analysis'].map(name_map)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
sns.barplot(
    data=plot_df, x='analysis_name', y='corr', hue='group',
    order=[name_map[k] for k in order], hue_order=['Real', 'Control'],
    estimator=np.mean, errorbar=('se', 1), capsize=0.08,
    palette={'Real': '#3b82f6', 'Control': '#9ca3af'},
    ax=ax
)

# paired t-test (Real vs Control) + significance brackets
from scipy import stats
def _p_to_star(p):
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'

pvals = {}
for ana in order:
    d = corr_detail_df[corr_detail_df['analysis'] == ana]
    _, p = stats.ttest_rel(d['corr_real'], d['corr_ctrl'], nan_policy='omit')
    pvals[ana] = p

real_bars, ctrl_bars = ax.containers[0], ax.containers[1]
# overlay paired lines (each real-control pair)
for i, ana in enumerate(order):
    d = corr_detail_df[corr_detail_df['analysis'] == ana]
    b1 = real_bars[i]
    b2 = ctrl_bars[i]
    x1 = b1.get_x() + b1.get_width() / 2
    x2 = b2.get_x() + b2.get_width() / 2
    msk = d['corr_real'].notna() & d['corr_ctrl'].notna()
    for y1, y2 in zip(d.loc[msk, 'corr_real'], d.loc[msk, 'corr_ctrl']):
        ax.plot([x1, x2], [y1, y2], color='k', alpha=0.14, lw=0.8, zorder=1)

for i, ana in enumerate(order):
    b1 = real_bars[i]
    b2 = ctrl_bars[i]
    x1 = b1.get_x() + b1.get_width() / 2
    x2 = b2.get_x() + b2.get_width() / 2
    y = max(b1.get_height(), b2.get_height()) + 0.06
    h = 0.025
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='k', lw=1)
    ax.text((x1 + x2) / 2, y + h + 0.005, _p_to_star(pvals[ana]),
            ha='center', va='bottom', fontsize=10)

ax.set_xlabel('')
ax.set_ylabel('Pearson r')
ax.set_ylim(-0.1, 1.1)
ax.grid(axis='y', alpha=0.2, lw=0.8)
ax.legend(frameon=False, title='')
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
fig.tight_layout()

#%% stimulus selection montages (set1_stims ↔ 132 pngs in stim_path)
import os
stim_path = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable\Stimset1'
fig_dir = ot.Join(save_path, 'Report_selections')
os.makedirs(fig_dir, exist_ok=True)

# Stimset / set1_stims col order: obj1..11, each obj = CRL-In (C,R,L) + CRL-Out (C,R,L)
# flat k within obj: 0,1,2=In-C,R,L; 3,4,5=Out-C,R,L
_K_META = [('In', 'C'), ('In', 'R'), ('In', 'L'), ('Out', 'C'), ('Out', 'R'), ('Out', 'L')]
_ST_OFF = {'Shading': 0, 'Texture': 66}

files = sorted(ot.Get_File_Name(stim_path, '.png'))
assert len(files) == 132, f'expected 132 png, got {len(files)}'
m_rsp = set1_stims.mean(0)

def _flat_col(obj, k, st):
    return _ST_OFF[st] + int(obj) * 6 + int(k)

def _parse_flat(col):
    col = int(col) % 66
    return col // 6, *_K_META[col % 6]

def _path_flat(col):
    return files[col]

def _save_grid(paths, title, fname, ncols=None):
    paths = list(paths)
    n = len(paths)
    ncols = ncols or min(n, 5)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, fp in zip(axes, paths):
        ax.imshow(plt.imread(fp))
        ax.set_title(os.path.basename(fp), fontsize=6)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(ot.Join(fig_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

def _inout_r_obj(obj, st='Shading'):
    """Per obj: corr(In-C,Out-C), corr(In-R,Out-R), corr(In-L,Out-L), then mean (no self-pairs)."""
    base = _ST_OFF[st] + int(obj) * 6
    cors = [_safe_corr(set1_stims[:, base + k], set1_stims[:, base + k + 3]) for k in range(3)]
    return np.nanmean(cors)

def _save_inout_c(obj, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))
    for ax, io in zip(axes, io_list):
        ax.imshow(plt.imread(_path_flat(_flat_col(obj, 0 if io == 'In' else 3, 'Shading'))))
        ax.set_title(f'{io}-C', fontsize=9)
        ax.axis('off')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(ot.Join(fig_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

summary_rows = []

# 1) 66 entities (flat stimset order); top-5 per Shading / Texture
for st in st_list:
    base = _ST_OFF[st]
    ents = [(m_rsp[base + c], c, *_parse_flat(c)) for c in range(66)]
    ents.sort(key=lambda x: x[0], reverse=True)
    top5 = ents[:5]
    for rank, (sc, c, o, io, ori) in enumerate(top5, 1):
        summary_rows.append({
            'analysis': 'top5_among_66', 'Shading/Texture': st, 'rank': rank,
            'Object': o + 1, 'In/Out': io, 'R-C-L': ori, 'mean_z': sc, 'col': base + c,
            'image': _path_flat(base + c),
        })
    _save_grid([_path_flat(base + c) for sc, c, o, io, ori in top5],
               f'{st} | top5 among 66', f'01_top5_66_{st}.png', ncols=5)

# 2) 22 entities (obj×In/Out, CLR averaged); top-3; display C only (k=0 In, k=3 Out)
for st in st_list:
    base = _ST_OFF[st]
    ents = []
    for o in range(_n_obj):
        for io, ks in [('In', (0, 1, 2)), ('Out', (3, 4, 5))]:
            sc = np.mean([m_rsp[base + o * 6 + k] for k in ks])
            ents.append((sc, o, io))
    ents.sort(key=lambda x: x[0], reverse=True)
    top3 = ents[:3]
    for rank, (sc, o, io) in enumerate(top3, 1):
        col = base + o * 6 + (0 if io == 'In' else 3)
        summary_rows.append({
            'analysis': 'top3_among_22', 'Shading/Texture': st, 'rank': rank,
            'Object': o + 1, 'In/Out': io, 'mean_z_clr_avg': sc, 'col': col,
            'image': _path_flat(col),
        })
    _save_grid([_path_flat(base + o * 6 + (0 if io == 'In' else 3)) for sc, o, io in top3],
               f'{st} | top3 among 22 (show C)', f'02_top3_22_{st}.png', ncols=3)

# 3) Shading only: In/Out similarity per obj (matched C/R/L pairs, then mean); top3 / bottom3
sim = []
for o in range(_n_obj):
    r = _inout_r_obj(o, 'Shading')
    sim.append((r, o))
sim.sort(key=lambda x: x[0], reverse=True)
for tag, lst in [('most', sim[:3]), ('least', sim[-3:][::-1])]:
    for rank, (r, o) in enumerate(lst, 1):
        summary_rows.append({
            'analysis': f'inout_{tag}_similar_shading', 'rank': rank,
            'Object': o + 1, 'InOut_r_clr_avg': r,
        })
        _save_inout_c(o, f'Shading | In–Out {tag} similar #{rank} (obj{o+1}, r={r:.2f})',
                      f'03_inout_{tag}_shading_obj{o+1}.png')

selection_summary = pd.DataFrame(summary_rows)
selection_summary.to_csv(ot.Join(fig_dir, 'selection_summary.csv'), index=False)
print(f'Saved {len(os.listdir(fig_dir))} files to: {fig_dir}')

#%% hierarchical corr heatmap: 2×2 (Shading/Texture) → 2×2 (In/Out) → 3×3 (R-C-L)
import itertools
# each cell = mean Pearson r across 11 objs (neuron-wise corr per obj, then average)
_blk = 6  # 2 In/Out × 3 R/C/L

def _k_io_ori(io, ori):
    return {'C': 0, 'R': 1, 'L': 2}[ori] + (0 if io == 'In' else 3)

def _mean_r_objs(st_a, io_a, ori_a, st_b, io_b, ori_b):
    ka, kb = _k_io_ori(io_a, ori_a), _k_io_ori(io_b, ori_b)
    cors = [_safe_corr(set1_stims[:, _flat_col(o, ka, st_a)], set1_stims[:, _flat_col(o, kb, st_b)])
            for o in range(_n_obj)]
    return np.nanmean(cors)

def _rc(st, io, ori):
    return st_list.index(st) * _blk + io_list.index(io) * 3 + ori_list.index(ori)

mat = np.zeros((2 * _blk, 2 * _blk))
for st_a, st_b in itertools.product(st_list, st_list):
    for io_a, io_b in itertools.product(io_list, io_list):
        for ori_a, ori_b in itertools.product(ori_list, ori_list):
            mat[_rc(st_a, io_a, ori_a), _rc(st_b, io_b, ori_b)] = _mean_r_objs(
                st_a, io_a, ori_a, st_b, io_b, ori_b)

mask = np.eye(mat.shape[0], dtype=bool)
_rc_l_lbl = ori_list * 4  # R,C,L × (Shading-In, Shading-Out, Texture-In, Texture-Out)
_ticks = np.arange(mat.shape[0]) + 0.5
_io_centers = [(1.5, 'In'), (4.5, 'Out'), (7.5, 'In'), (10.5, 'Out')]
_st_centers = [(3, 'Shading'), (9, 'Texture')]

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(mat, mask=mask, vmin=-0.5, vmax=0.5, center=0, cmap='RdBu_r',
            xticklabels=_rc_l_lbl, yticklabels=_rc_l_lbl, square=True, linewidths=0,
            cbar_kws={'label': 'mean r (11 objs)', 'shrink': 0.8}, ax=ax)
ax.tick_params(axis='both', labelsize=8, length=0, pad=1)
for x in (3, 9):
    ax.axvline(x, color='gold', lw=1.5, zorder=3)
    ax.axhline(x, color='gold', lw=1.5, zorder=3)
ax.axvline(6, color='k', lw=2, zorder=3)
ax.axhline(6, color='k', lw=2, zorder=3)
for xc, lab in _io_centers:
    ax.text(xc, -0.055, lab, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)
    ax.text(-0.055, xc, lab, ha='right', va='center', transform=ax.get_yaxis_transform(), fontsize=9, rotation=90)
for xc, lab in _st_centers:
    ax.text(xc, -0.105, lab, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=10)
    ax.text(-0.105, xc, lab, ha='right', va='center', transform=ax.get_yaxis_transform(), fontsize=10, rotation=90)
fig.subplots_adjust(bottom=0.10, left=0.10)
fig.savefig(ot.Join(fig_dir, '04_hierarchical_corr_heatmap.png'), dpi=150, bbox_inches='tight', pad_inches=0.05)

#%%


