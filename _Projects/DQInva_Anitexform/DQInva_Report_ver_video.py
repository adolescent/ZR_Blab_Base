'''
Some analysis for the video data.
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
set2_video = np.load(ot.Join(save_path,'dq_video_BFCT.npy'),allow_pickle=True)

#%% RSA: 78×78 (19 Body | 20 Face | 19 Chair | 20 Tool)
rsa = np.corrcoef(set2_video.T)
np.fill_diagonal(rsa, 0)

_cat_divs = [19, 39, 58]
_cat_lbl = ['Body', 'Face', 'Chair', 'Tool']
_cat_ctr = [9, 29, 48, 68]

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(rsa, vmin=-0.5, vmax=0.5, center=0, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, square=True,
            cbar_kws={'shrink': 0.7, 'label': 'Pearson r'}, ax=ax,cbar=False)
for x in _cat_divs:
    ax.axvline(x, color='yellow', lw=2)
    ax.axhline(x, color='yellow', lw=2)
for c, lbl in zip(_cat_ctr, _cat_lbl):
    ax.text(c, 1.04, lbl, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=9, fontweight='bold')
fig.tight_layout()

#%% Within vs between (split by other category): 1 within + 3 between per anchor
from scipy import stats

_cat_sizes = [19, 20, 19, 20]
_cat_id = np.repeat(np.arange(4), _cat_sizes)
_other = lambda c: [oc for oc in range(4) if oc != c]

# rsa_pairs[anchor][other] = 1d corrs; rsa_pairs[anchor][anchor] = within
rsa_pairs = {}
for c in range(4):
    idx = np.where(_cat_id == c)[0]
    rsa_pairs[c] = {c: rsa[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)]}
    for oc in _other(c):
        oidx = np.where(_cat_id == oc)[0]
        rsa_pairs[c][oc] = rsa[np.ix_(idx, oidx)].ravel()

rsa_summary = pd.DataFrame([
    dict(anchor=_cat_lbl[c], pair='within' if oc == c else f'{_cat_lbl[c]}–{_cat_lbl[oc]}',
         mean=rsa_pairs[c][oc].mean(), sem=rsa_pairs[c][oc].std(ddof=1) / np.sqrt(len(rsa_pairs[c][oc])),
         n=len(rsa_pairs[c][oc]))
    for c in range(4) for oc in [c] + _other(c)
])
rsa_tests = pd.DataFrame([
    dict(anchor=_cat_lbl[c], comparison=f'within > {_cat_lbl[oc]}',
         p=stats.mannwhitneyu(rsa_pairs[c][c], rsa_pairs[c][oc], alternative='greater').pvalue)
    for c in range(4) for oc in _other(c)
])

def _p_to_star(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'

def _sig_bracket(ax, x1, x2, y, h, text, fs=9):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='k', lw=0.9, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=fs)

_col_in, _col_cross = '#4C72B0', '#C4C4C4'
_n_bar = 4
_bw = 0.18
_gap = 0.22
_offsets = (np.arange(_n_bar) - (_n_bar - 1) / 2) * _gap
_group_x = np.arange(4)

_plot = []
for c in range(4):
    ocs = [c] + _other(c)
    vals = [rsa_pairs[c][oc] for oc in ocs]
    means = np.array([v.mean() for v in vals])
    sems = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in vals])
    xs = _group_x[c] + _offsets
    bar_lbl = ['In'] + [_cat_lbl[oc] for oc in _other(c)]
    tests = [(0, k, stats.mannwhitneyu(vals[0], vals[k], alternative='greater').pvalue)
             for k in range(1, 4)]
    _plot.append(dict(xs=xs, means=means, sems=sems, bar_lbl=bar_lbl, tests=tests))

fig, ax = plt.subplots(figsize=(7, 5.5))
for d in _plot:
    ax.bar(d['xs'], d['means'], width=_bw, yerr=d['sems'],
           color=[_col_in] + [_col_cross] * 3, capsize=2.5, error_kw={'linewidth': 0.9})
    for x, lb in zip(d['xs'], d['bar_lbl']):
        ax.text(x, -0.055, lb, ha='center', va='top', fontsize=7,
                transform=ax.get_xaxis_transform())

_yspan = max((d['means'] + d['sems']).max() for d in _plot) - min(
    (d['means'] - d['sems']).min() for d in _plot)
_dy = max(0.04, 0.09 * _yspan)
_bh = _dy * 0.2
for d in _plot:
    y0 = (d['means'] + d['sems']).max() + 0.02 * _yspan
    for k, (_, j, p) in enumerate(d['tests'], start=1):
        _sig_bracket(ax, d['xs'][0], d['xs'][j], y0 + (k - 1) * _dy, _bh, _p_to_star(p))

ax.set_xticks(_group_x)
ax.set_xticklabels(_cat_lbl, fontsize=11)
ax.set_ylabel('Pearson r')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=_col_in, label='Within'),
                   Patch(facecolor=_col_cross, label='Cross-category')],
          frameon=False, loc='upper right')
# ax.text(0.01, 0.98, 'Brackets: within vs cross (Mann–Whitney, one-sided)\n* p<0.05, ** p<0.01, *** p<0.001',transform=ax.transAxes, va='top', fontsize=7)
_ylim = ax.get_ylim()
ax.set_ylim(_ylim[0], _ylim[1] + 3.2 * _dy)
fig.tight_layout()

#%% LOO linear SVM: decode category (78 obj samples × population)
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = set2_video.T
y = np.repeat(np.arange(4), _cat_sizes)

clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))
_loo = LeaveOneOut()
loo_acc = cross_val_score(clf, X, y, cv=_loo, scoring='accuracy').mean()
y_pred = cross_val_predict(clf, X, y, cv=_loo)
loo_per_class = pd.DataFrame([
    dict(category=_cat_lbl[c],
         n=(y == c).sum(),
         accuracy=(y_pred[y == c] == c).mean())
    for c in range(4)
])
print(loo_per_class.to_string(index=False, formatters={'accuracy': '{:.3f}'.format}))

_rng = np.random.default_rng(0)
_shuffle_acc = np.array([
    cross_val_score(clf, X, _rng.permutation(y), cv=_loo, scoring='accuracy').mean()
    for _ in range(10)
])
print(f'LOO accuracy: {loo_acc:.3f}  (chance = 0.25)')
print(f'Label-shuffle control (n=10): {_shuffle_acc.mean():.3f} ± {_shuffle_acc.std():.3f}')

#%%

