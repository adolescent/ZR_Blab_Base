'''

Test response of texture transformed object .
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
save_path = r'E:\#Preprocessed_Data\Selected_Cells\Ani_Texform'

site = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\ani_texform','.joblib')[0]
a = JL.load(site)

#%% Load data, select good cells.
stim_info = a.stim_info
ani_cells,ani_psth = a.Cell_Selection(ceiling=0.2,prefer='Animate',dp_thres=0.5)
len(ani_cells)
redplot = ani_psth[:,72:,150:320].sum(-1)
redplot_z = (redplot-redplot.mean(1,keepdims=True))/redplot.std(1,keepdims=True)
np.savez_compressed(ot.Join(save_path,'Ani_Texform_Redplot_z.npz'),redplot_z=redplot_z)
#%% heatmap: redplot_z = Texform_Obj only (480 stims)
info_tex = stim_info.iloc[72:].reset_index(drop=True)
cat_divs = info_tex.groupby(info_tex.Category.ne(info_tex.Category.shift()).cumsum()).size().cumsum().iloc[:-1].tolist()
con_div = 240                    # LowCon | HighCon
texform_divs = [120, 360]        # Real | Texform within each Con block

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(redplot_z, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'z-scored response', 'shrink': 0.8}, ax=ax)
for x in cat_divs:
    ax.axvline(x, color='yellow', lw=1.5)
ax.axvline(con_div, color='black', lw=2)
for x in texform_divs:
    ax.axvline(x, color='C0', lw=2)
ax.set_xlabel('LowCon Real  |  LowCon Texform  |  HighCon Real  |  HighCon Texform')
ax.set_ylabel('Neuron')
fig.tight_layout()

#%% per-object Pearson r across conditions (120 objs × 4 pairs × matched/control)
R = redplot_z
info_tex = stim_info.iloc[72:].reset_index(drop=True)
obj_meta = info_tex.iloc[:120][['Object', 'Category']].drop_duplicates('Object')
obj_meta['Subtype'] = obj_meta.Category.str.replace(r'_(LowCon|HighCon|Texform_LowCon|Texform_HighCon)$', '', regex=True)
subtype_objs = obj_meta.groupby('Subtype')['Object'].apply(lambda s: sorted(s.tolist())).to_dict()

pairs = [
    ('HighCon-LowCon', lambda o: (240 + o, o)),
    ('HighCon-HighCon_Texform', lambda o: (240 + o, 360 + o)),
    ('LowCon-LowCon_Texform', lambda o: (o, 120 + o)),
    ('HighCon_Texform-LowCon_Texform', lambda o: (360 + o, 120 + o)),
]
rows = []
for _, row in obj_meta.iterrows():
    obj_id = int(row.Object)
    o = obj_id - 1
    others = [j - 1 for j in subtype_objs[row.Subtype] if j != obj_id]
    for corr_type, idx_fn in pairs:
        c1, c2 = idx_fn(o)
        rows.append({
            'Object': obj_id, 'Subtype': row.Subtype, 'Corr_type': corr_type,
            'Match': 'matched', 'Corr': np.corrcoef(R[:, c1], R[:, c2])[0, 1],
        })
        ctrl = [np.corrcoef(R[:, idx_fn(o)[0]], R[:, idx_fn(j)[1]])[0, 1] for j in others]
        rows.append({
            'Object': obj_id, 'Subtype': row.Subtype, 'Corr_type': corr_type,
            'Match': 'control', 'Corr': np.mean(ctrl),
        })
obj_cond_corr = pd.DataFrame(rows)

#%% connected bar plot + paired t-test (x = Subtype, hue = Corr_type)
from scipy import stats
from matplotlib.patches import Patch

def p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''

def sig_bracket(ax, x1, x2, y, stars, h=0.015):
    if not stars:
        return y
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], 'k-', lw=1)
    ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom', fontsize=11)
    return y + h + 0.04

corr_order = ['HighCon-LowCon', 'HighCon-HighCon_Texform', 'LowCon-LowCon_Texform',
              'HighCon_Texform-LowCon_Texform']
corr_short = ['H-L', 'H-T', 'L-T', 'Tx H-L']
subtypes = ['Ani_Big', 'Obj_Big', 'Ani_Small', 'Obj_Small']
colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']
x0 = np.arange(4)
w = 0.16
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w
pair_idx = [(a, b) for a in range(4) for b in range(a + 1, 4)]

fig, ax = plt.subplots(figsize=(12, 5))
y_max = 0
dw, bw = w * 0.22, w * 0.38
for i, sub in enumerate(subtypes):
    d = obj_cond_corr.query('Subtype == @sub')
    wide = d.query('Match == "matched"').pivot(index='Object', columns='Corr_type', values='Corr')[corr_order]
    wide_ctrl = d.query('Match == "control"').pivot(index='Object', columns='Corr_type', values='Corr')[corr_order]
    x_ctr = x0[i] + offsets - dw / 2
    x_mat = x0[i] + offsets + dw / 2
    ax.bar(x_ctr, wide_ctrl.mean(), yerr=wide_ctrl.sem(), width=bw, capsize=3,
           color=colors, alpha=0.35, edgecolor=colors, linewidth=0.8)
    ax.bar(x_mat, wide.mean(), yerr=wide.sem(), width=bw, capsize=3, color=colors)
    for _, row in wide.iterrows():
        ax.plot(x_mat, row.values, 'k-', lw=0.6, alpha=0.35)
    y_base = max((wide.mean() + wide.sem()).max(), (wide_ctrl.mean() + wide_ctrl.sem()).max()) + 0.02
    for k, (a, b) in enumerate(pair_idx):
        p = stats.ttest_rel(wide[corr_order[a]], wide[corr_order[b]]).pvalue
        sig_bracket(ax, x_mat[a], x_mat[b], y_base + k * 0.07, p_to_stars(p))
    y_max = max(y_max, y_base + (len(pair_idx) - 1) * 0.07 + 0.06)

ax.set_xticks(x0, subtypes)
ax.set_ylabel('Pearson r')
ax.set_ylim(top=max(y_max, ax.get_ylim()[1]))
ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, corr_short)]
          + [Patch(facecolor='gray', alpha=0.35, label='Control'),
             Patch(facecolor='gray', label='Matched')],
          title='Corr type', loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
fig.tight_layout(rect=[0, 0, 0.86, 1])

#%% SVM cross-decoding with split-half pseudotrials
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

a = JL.load(site)
rsp_by_trail = a.raw_psth[ani_cells, :, 72:, 160:320].sum(-1)  # (cell, trial, stim)
n_cell, n_trial, n_stim = rsp_by_trail.shape
assert n_stim == 480 and n_trial == 24

conds = ['LowCon_Real', 'LowCon_Texform', 'HighCon_Real', 'HighCon_Texform']
cond_off = [0, 120, 240, 360]
subtypes = ['Ani_Big', 'Obj_Big', 'Ani_Small', 'Obj_Small']
sub_off = [0, 30, 60, 90]
n_class = 30
chance = 1 / n_class
n_splits = 5

def build_pseudotrials(rsp, cond_start, sub_start, seed=0):
    """10 pseudotrials/class: 5 random split-half averages, store trial masks."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_splits):
        perm = rng.permutation(n_trial)
        h1, h2 = perm[:12], perm[12:]
        m1 = np.zeros(n_trial, dtype=bool); m1[h1] = True
        m2 = np.zeros(n_trial, dtype=bool); m2[h2] = True
        for c in range(n_class):
            col = cond_start + sub_start + c
            samples.append({'X': rsp[:, h1, col].mean(1), 'y': c, 'mask': m1})
            samples.append({'X': rsp[:, h2, col].mean(1), 'y': c, 'mask': m2})
    return samples

def loo_same_cond(samples):
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    pred, true = [], []
    for i, te in enumerate(samples):
        tr = [s for j, s in enumerate(samples)
              if j != i and not np.any(s['mask'] & te['mask'])]
        if len(tr) < n_class:
            continue
        X_tr = np.vstack([s['X'] for s in tr])
        y_tr = np.array([s['y'] for s in tr])
        if np.unique(y_tr).size < n_class:
            continue
        clf.fit(X_tr, y_tr)
        pred.append(clf.predict(te['X'].reshape(1, -1))[0])
        true.append(te['y'])
    return np.mean(np.array(pred) == np.array(true)) if pred else np.nan

def cross_cond(train_samples, test_samples):
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    X_tr = np.vstack([s['X'] for s in train_samples])
    y_tr = np.array([s['y'] for s in train_samples])
    X_te = np.vstack([s['X'] for s in test_samples])
    y_te = np.array([s['y'] for s in test_samples])
    clf.fit(X_tr, y_tr)
    return (clf.predict(X_te) == y_te).mean()

# precompute pseudotrials: pseudo[(sub, cond)] -> list of samples
pseudo = {}
for si, sub in enumerate(subtypes):
    for ci, cond in enumerate(conds):
        pseudo[(sub, cond)] = build_pseudotrials(rsp_by_trail, cond_off[ci], sub_off[si], seed=si * 10 + ci)

rows = []
for sub in subtypes:
    for tr_cond in conds:
        tr_s = pseudo[(sub, tr_cond)]
        for te_cond in conds:
            te_s = pseudo[(sub, te_cond)]
            if tr_cond == te_cond:
                acc = loo_same_cond(tr_s)
                mode = 'LOO_same_condition'
            else:
                acc = cross_cond(tr_s, te_s)
                mode = 'cross_condition'
            rows.append({
                'Subtype': sub, 'Train_cond': tr_cond, 'Test_cond': te_cond,
                'Accuracy': acc, 'N_class': n_class,
                'N_train_samples': len(tr_s), 'N_test_samples': len(te_s),
                'Eval_mode': mode, 'Chance': chance,
            })
decode_df = pd.DataFrame(rows)
# Per-subtype 4x4 decode matrices
decode_pivot = {
    sub: (decode_df.loc[decode_df['Subtype'] == sub]
          .pivot(index='Train_cond', columns='Test_cond', values='Accuracy')
          .reindex(index=conds, columns=conds))
    for sub in subtypes
}
# Optional overall mean matrix across subtypes
decode_pivot_mean = (decode_df
                     .pivot_table(index='Train_cond', columns='Test_cond',
                                  values='Accuracy', aggfunc='mean')
                     .reindex(index=conds, columns=conds))

#%% bootstrap label-shuffle control (chance baseline)
def loo_same_cond_shuffle(samples, rng):
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    pred, true = [], []
    for i, te in enumerate(samples):
        tr = [s for j, s in enumerate(samples)
              if j != i and not np.any(s['mask'] & te['mask'])]
        if len(tr) < n_class:
            continue
        X_tr = np.vstack([s['X'] for s in tr])
        y_tr = np.array([s['y'] for s in tr])
        if np.unique(y_tr).size < n_class:
            continue
        y_tr = rng.permutation(y_tr)  # shuffle labels only
        clf.fit(X_tr, y_tr)
        pred.append(clf.predict(te['X'].reshape(1, -1))[0])
        true.append(te['y'])
    return np.mean(np.array(pred) == np.array(true)) if pred else np.nan

def cross_cond_shuffle(train_samples, test_samples, rng):
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    X_tr = np.vstack([s['X'] for s in train_samples])
    y_tr = np.array([s['y'] for s in train_samples])
    X_te = np.vstack([s['X'] for s in test_samples])
    y_te = np.array([s['y'] for s in test_samples])
    clf.fit(X_tr, rng.permutation(y_tr))  # shuffle labels only
    return (clf.predict(X_te) == y_te).mean()

n_boot = 10  # increase if you want tighter control CI
rng_boot = np.random.default_rng(0)
rows_ctrl = []
for b in tqdm(range(n_boot)):
    for sub in subtypes:
        for tr_cond in conds:
            tr_s = pseudo[(sub, tr_cond)]
            for te_cond in conds:
                te_s = pseudo[(sub, te_cond)]
                if tr_cond == te_cond:
                    acc = loo_same_cond_shuffle(tr_s, rng_boot)
                    mode = 'LOO_same_condition'
                else:
                    acc = cross_cond_shuffle(tr_s, te_s, rng_boot)
                    mode = 'cross_condition'
                rows_ctrl.append({
                    'Bootstrap': b,
                    'Subtype': sub,
                    'Train_cond': tr_cond,
                    'Test_cond': te_cond,
                    'Accuracy': acc,
                    'Eval_mode': mode,
                })

decode_df_ctrl = pd.DataFrame(rows_ctrl)
decode_df_ctrl_mean = (decode_df_ctrl
                       .groupby(['Subtype', 'Train_cond', 'Test_cond', 'Eval_mode'], as_index=False)['Accuracy']
                       .mean())
decode_pivot_ctrl_mean = (decode_df_ctrl
                          .pivot_table(index='Train_cond', columns='Test_cond',
                                       values='Accuracy', aggfunc='mean')
                          .reindex(index=conds, columns=conds))

#%% summary heatmap of decoding accuracy (diag = LOO)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
sns.heatmap(
    decode_pivot_mean,
    ax=ax,
    cmap='RdYlBu_r',
    vmin=max(0, chance),
    vmax=max(float(decode_pivot_mean.max().max()), chance + 0.02),
    annot=True,
    fmt='.2f',
    linewidths=0.6,
    square=True,
    cbar_kws={'label': 'Accuracy', 'shrink': 0.85},
)
ax.set_title('Cross-decoding summary (diag = LOO)')
ax.set_xlabel('Test condition')
ax.set_ylabel('Train condition')
fig.tight_layout()

#%% side-by-side heatmaps: real vs shuffle-control (subtype selectable)
# plot_subtype: 'mean' | single subtype | list e.g. ['Ani_Big', 'Ani_Small']
# plot_subtype = ['Ani_Big', 'Ani_Small']
# plot_subtype = ['Ani_Small']
# plot_subtype = ['Obj_Big', 'Obj_Small']  
plot_subtype = ['Obj_Small']
plot_order = ['LowCon_Real', 'HighCon_Real', 'LowCon_Texform', 'HighCon_Texform']
tick_short = ['L-Real', 'H-Real', 'L-Tex', 'H-Tex']

if plot_subtype == 'mean':
    sel = subtypes
elif isinstance(plot_subtype, str):
    sel = [plot_subtype]
else:
    sel = list(plot_subtype)
plot_label = 'mean' if plot_subtype == 'mean' else '+'.join(sel)

real_src = (decode_df.loc[decode_df['Subtype'].isin(sel)]
            .pivot_table(index='Train_cond', columns='Test_cond',
                         values='Accuracy', aggfunc='mean')
            .reindex(index=conds, columns=conds))
ctrl_src = (decode_df_ctrl.loc[decode_df_ctrl['Subtype'].isin(sel)]
            .pivot_table(index='Train_cond', columns='Test_cond',
                         values='Accuracy', aggfunc='mean')
            .reindex(index=conds, columns=conds))

real_plot = real_src.reindex(index=plot_order, columns=plot_order)
ctrl_plot = ctrl_src.reindex(index=plot_order, columns=plot_order)
vmax_pair = max(float(real_plot.max().max()), float(ctrl_plot.max().max()), chance + 0.02)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

sns.heatmap(
    real_plot,
    ax=axes[0],
    cmap='YlOrRd',
    vmin=chance,
    vmax=vmax_pair,
    annot=True,
    fmt='.2f',
    linewidths=0.6,
    square=True,
    cbar=False,
)
axes[0].set_title(f'Real decoding ({plot_label}, diag = LOO)')
axes[0].set_xlabel('Test condition')
axes[0].set_ylabel('Train condition')
axes[0].set_xticklabels(tick_short, rotation=0)
axes[0].set_yticklabels(tick_short, rotation=0)

hm = sns.heatmap(
    ctrl_plot,
    ax=axes[1],
    cmap='YlOrRd',
    vmin=chance,
    vmax=vmax_pair,
    annot=True,
    fmt='.2f',
    linewidths=0.6,
    square=True,
    cbar_kws={'label': 'Accuracy (origin = chance 0.033)', 'shrink': 0.85,
              'ticks': np.linspace(chance, vmax_pair, 5)},
)
axes[1].set_title(f'Shuffle-label control ({plot_label})')
axes[1].set_xlabel('Test condition')
axes[1].set_ylabel('')
axes[1].set_xticklabels(tick_short, rotation=0)
axes[1].set_yticklabels(tick_short, rotation=0)

fig.tight_layout()
#%%

