'''
Try cross decoding between different process to graphs.
'''

#%% Load data

from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

save_path = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable'

site = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\DQInva', '.joblib')[0]
a = JL.load(site)
ani_cells, ani_psth = a.Cell_Selection(ceiling=0.2, prefer='Animate', dp_thres=0.5)
ani_psth_bytrail = a.raw_psth[ani_cells, :, 360:, 150:500].sum(-1)  # (cell, trial, 480)

# Average duplicate presentations: cols 0–239 and 240–479 are identical blocks
rsp = (ani_psth_bytrail[..., :240] + ani_psth_bytrail[..., 240:]) / 2
n_cell, n_trial, n_stim = rsp.shape
assert n_stim == 240

#%% Cross-decoding: split-half pseudotrials + LOO / cross-condition SVM

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SUBCATS = ['Body', 'Face', 'Fruit']
SUB_OFF = [0, 80, 160]
CONDS = ['Tex_CTR', 'Shading_CTR', 'Tex', 'Shading']
COND_IDX = [0, 1, 2, 3]
N_CLASS = 20
CHANCE = 1 / N_CLASS
N_SPLITS = 5
half = n_trial // 2


def stim_col(sub_off, obj_i, cond_i):
    """Column index in rsp for one stimulus (obj 0–19, cond 0–3)."""
    return sub_off + obj_i * 4 + cond_i


def build_pseudotrials(rsp, sub_off, cond_i, seed=0):
    """40 pseudotrials/class: 5 split-half pairs × 2 halves × 20 objects."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(N_SPLITS):
        perm = rng.permutation(n_trial)
        h1, h2 = perm[:half], perm[half:]
        m1 = np.zeros(n_trial, dtype=bool)
        m1[h1] = True
        m2 = np.zeros(n_trial, dtype=bool)
        m2[h2] = True
        for c in range(N_CLASS):
            col = stim_col(sub_off, c, cond_i)
            samples.append({'X': rsp[:, h1, col].mean(1), 'y': c, 'mask': m1})
            samples.append({'X': rsp[:, h2, col].mean(1), 'y': c, 'mask': m2})
    return samples


def loo_same_cond(samples):
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    pred, true = [], []
    for i, te in enumerate(samples):
        tr = [s for j, s in enumerate(samples)
              if j != i and not np.any(s['mask'] & te['mask'])]
        if len(tr) < N_CLASS:
            continue
        X_tr = np.vstack([s['X'] for s in tr])
        y_tr = np.array([s['y'] for s in tr])
        if np.unique(y_tr).size < N_CLASS:
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


pseudo = {}
for si, sub in enumerate(SUBCATS):
    for ci, cond in enumerate(CONDS):
        pseudo[(sub, cond)] = build_pseudotrials(rsp, SUB_OFF[si], COND_IDX[ci], seed=si * 10 + ci)

rows = []
for sub in SUBCATS:
    for tr_cond in CONDS:
        tr_s = pseudo[(sub, tr_cond)]
        for te_cond in CONDS:
            te_s = pseudo[(sub, te_cond)]
            if tr_cond == te_cond:
                acc = loo_same_cond(tr_s)
                mode = 'LOO_same_condition'
            else:
                acc = cross_cond(tr_s, te_s)
                mode = 'cross_condition'
            rows.append({
                'Subclass': sub,
                'Train_cond': tr_cond,
                'Test_cond': te_cond,
                'Accuracy': acc,
                'N_class': N_CLASS,
                'N_train_samples': len(tr_s),
                'N_test_samples': len(te_s),
                'Eval_mode': mode,
                'Chance': CHANCE,
            })
decode_df = pd.DataFrame(rows)

decode_pivot = {
    sub: (decode_df.loc[decode_df['Subclass'] == sub]
          .pivot(index='Train_cond', columns='Test_cond', values='Accuracy')
          .reindex(index=CONDS, columns=CONDS))
    for sub in SUBCATS
}
decode_pivot_mean = (decode_df
                     .pivot_table(index='Train_cond', columns='Test_cond',
                                  values='Accuracy', aggfunc='mean')
                     .reindex(index=CONDS, columns=CONDS))

#%% Heatmaps

plot_order = ['Shading', 'Tex', 'Shading_CTR', 'Tex_CTR']
COND_SHORT = ['Sh', 'Tx', 'SC', 'TC']
vmin, vmax = 0, max(float(decode_df['Accuracy'].max()), 0.07)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, sub in zip(axes, SUBCATS):
    mat = decode_pivot[sub].reindex(index=plot_order, columns=plot_order)
    sns.heatmap(
        mat, ax=ax, cmap='RdBu_r', vmin=vmin, vmax=vmax,
        annot=True, fmt='.2f', linewidths=0.6, square=True, cbar=False,center=0.05
    )
    ax.set_title(f'{sub} (diag = LOO)')
    ax.set_xlabel('Test')
    ax.set_ylabel('Train')
    ax.set_xticklabels(COND_SHORT, rotation=0)
    ax.set_yticklabels(COND_SHORT, rotation=0)
fig.suptitle('Cross-decoding by subclass', y=1.02)
fig.tight_layout()

fig, ax = plt.subplots(figsize=(5.5, 4.8))
sns.heatmap(
    decode_pivot_mean.reindex(index=plot_order, columns=plot_order),
    ax=ax, cmap='RdBu_r', vmin=vmin, vmax=vmax,
    annot=True, fmt='.2f', linewidths=0.6, square=True,
    cbar_kws={'label': 'Accuracy', 'shrink': 0.85, 'ticks': np.linspace(vmin, vmax, 5)},center=0.05
)
ax.set_title('Mean across Body / Face / Fruit (diag = LOO)')
ax.set_xlabel('Test condition')
ax.set_ylabel('Train condition')
ax.set_xticklabels(COND_SHORT, rotation=0)
ax.set_yticklabels(COND_SHORT, rotation=0)
fig.tight_layout()

#%%
