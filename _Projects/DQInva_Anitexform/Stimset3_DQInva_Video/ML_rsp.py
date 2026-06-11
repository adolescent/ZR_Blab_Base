
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

video_site_path = r'E:\#Preprocessed_Data\SiteClass\DQInva_video'
_ceiling, _dp_thres = 0.2, 0.5
_video_t_slice = slice(1050, 2500)   # -1000~3500 ms (video site alignment)
_n_fob = 72
_n_cycle, _n_bfct = 3, 78
_fob_divs = [24, 48]
_cat_sizes = [19, 20, 19, 20]
_cat_divs = [19, 39, 58]
_cat_lbl = ['Body', 'Face', 'Chair', 'Tool']
_bfct_starts = np.cumsum([0] + _cat_sizes[:-1])
_cat_ctr = [_bfct_starts[i] + _cat_sizes[i] / 2 for i in range(4)]  # heatmap column centers

#%% extract video responses (ML / face-prefer cells)
site = ot.Get_File_Name(video_site_path, '.joblib')[0]
a = JL.load(site)
ml_cells, ml_psth = a.Cell_Selection(ceiling=_ceiling, prefer='face', dp_thres=_dp_thres)

# FOB: raw_psth cols 0:72 = 24 Body | 24 Face | 24 Object
fob_rsp = a.raw_psth[ml_cells][:, :, :_n_fob, _video_t_slice].sum(-1).mean(1)
fob_rsp_z = (fob_rsp - fob_rsp.mean(1, keepdims=True)) / fob_rsp.std(1, keepdims=True)

redplot = ml_psth[:, :, _video_t_slice].sum(-1)
redplot_z = (redplot - redplot.mean(1, keepdims=True)) / redplot.std(1, keepdims=True)
video_rsp_z = redplot_z[:, _n_fob:].reshape(len(redplot_z), _n_cycle, _n_bfct).mean(1)
n_cell = video_rsp_z.shape[0]

#%% FOB heatmap
fig, ax = plt.subplots(figsize=(3, 6))
sns.heatmap(fob_rsp_z, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar=False, ax=ax)
for x in _fob_divs:
    ax.axvline(x, color='yellow', lw=2)
ax.set_title('FOB Response')
ax.set_xlabel('Body    |    Face    |  Object')
ax.set_ylabel(f'N_Cell={n_cell}')
fig.tight_layout()
plt.show()

#%% Category PSTH: avg objects × 3 cycles → Body / Face / Chair / Tool
bin_ms = 30
_t_onset = 1000
_t_plot_lo, _t_plot_hi = -400, 2000
t_ms = np.arange(ml_psth.shape[-1]) - _t_onset
msk = (t_ms >= _t_plot_lo) & (t_ms <= _t_plot_hi)

bfct_psth = ml_psth[:, _n_fob:, :][:, :, msk]
bfct_psth = bfct_psth.reshape(n_cell, _n_cycle, _n_bfct, -1).mean(1)  # avg 3 cycles

_cat_slices = [slice(0, 19), slice(19, 39), slice(39, 58), slice(58, 78)]
n_t = bfct_psth.shape[-1] // bin_ms * bin_ms
t_plot = t_ms[msk][:n_t].reshape(-1, bin_ms).mean(1)
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(50, 1500, color='lightgreen', alpha=0.2, zorder=0)  # redplot window 1050:2500
ax.axvline(0, color='gray', ls='--', lw=1)
for lbl, c, s in zip(_cat_lbl, colors, _cat_slices):
    obj_avg = bfct_psth[:, s, :].mean(1)  # avg over objects in category
    fr = obj_avg[..., :n_t].reshape(n_cell, -1, bin_ms).mean(-1) * 1000
    y = fr.mean(0)
    err = fr.std(0, ddof=1) / np.sqrt(n_cell)
    ax.plot(t_plot, y, lw=2, color=c, label=lbl)
    ax.fill_between(t_plot, y - err, y + err, color=c, alpha=0.2, linewidth=0)
ax.set_xlim(_t_plot_lo, _t_plot_hi)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate (Hz)')
ax.set_title('Category-averaged response of ML neurons')
ax.legend(frameon=False, loc='best')
fig.tight_layout()
plt.show()

#%% Video redplot: Body | Face | Chair | Tool (19 + 20 + 19 + 20)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(video_rsp_z, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar=False, ax=ax)
for x in _cat_divs:
    ax.axvline(x, color='yellow', lw=2)
for c, lbl in zip(_cat_ctr, _cat_lbl):
    ax.text(c, -0.05, lbl, transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=9, fontweight='bold')
ax.set_ylabel(f'N_Cell={n_cell}')
ax.set_title('Red plot of ML neurons, z-scored')
fig.tight_layout()
fig.subplots_adjust(bottom=0.14)
plt.show()

#%% RSA (same order as redplot above): 78×78 Pearson r
rsa = np.corrcoef(video_rsp_z.T)
np.fill_diagonal(rsa, 0)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(rsa, vmin=-0.7, vmax=0.7, center=0, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, square=True, cbar=False, ax=ax)
for x in _cat_divs:
    ax.axvline(x, color='yellow', lw=2)
    ax.axhline(x, color='yellow', lw=2)
for c, lbl in zip(_cat_ctr, _cat_lbl):
    ax.text(c, 1.04, lbl, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('RSA of ML neurons (Pearson r)')
fig.tight_layout()
plt.show()

#%% LOO 4-class SVM: trial-averaged population → Body / Face / Chair / Tool
# Each fold leaves one object out; SVM trains on the other 77 objects' population
# vectors, then decodes the held-out object's category.
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

_decode_t_slice = _video_t_slice   # scalar sum window; change to use another time frame


def _obj_pop_vectors(psth, t_slice):
    """(cell, stim, time) → (n_obj, n_cell) trial-averaged population per object."""
    rsp = psth[:, _n_fob:, t_slice].sum(-1)
    rsp = rsp.reshape(n_cell, _n_cycle, _n_bfct).mean(1)  # avg 3 cycles
    return rsp.T


X = _obj_pop_vectors(ml_psth, _decode_t_slice)
y = np.repeat(np.arange(4), _cat_sizes)

clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))
_loo = LeaveOneOut()
loo_acc = cross_val_score(clf, X, y, cv=_loo, scoring='accuracy').mean()
y_pred = cross_val_predict(clf, X, y, cv=_loo)

loo_per_class = pd.DataFrame([
    dict(category=_cat_lbl[c],
         n=int((y == c).sum()),
         accuracy=float((y_pred[y == c] == c).mean()))
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

cm = confusion_matrix(y, y_pred)
cm_prop = cm / cm.sum(axis=1, keepdims=True)  # row-normalize: proportion per true class
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_prop, annot=True, fmt='.0%', vmin=0, vmax=1, cmap='Blues',
            square=True, cbar=False, ax=ax,
            xticklabels=_cat_lbl, yticklabels=_cat_lbl)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'LOO confusion matrix — ML neurons (acc={loo_acc:.2f})')
fig.tight_layout()
plt.show()

#%% Cross-time LOO decoding: train SVM in one epoch, test in another
# Value = decode accuracy. x = train window, y = test window; diagonal = LOO.
_cross_win_ms = 100
_cross_step_ms = 100
_cross_t_lo, _cross_t_hi = 100, 2000   # ms relative to stimulus onset
_chance = 0.25

bfct_avg = ml_psth[:, _n_fob:, :].reshape(n_cell, _n_cycle, _n_bfct, -1).mean(1)
t_ms_full = np.arange(ml_psth.shape[-1]) - _t_onset
y_cat = np.repeat(np.arange(4), _cat_sizes)


def _time_windows(t_lo, t_hi, win_ms, step_ms):
    """Epoch list → (start_ms, end_ms) pairs and window-center labels."""
    wins, ctrs = [], []
    t = t_lo
    while t + win_ms <= t_hi + 1e-9:
        wins.append((t, t + win_ms))
        ctrs.append(int(t + win_ms / 2))
        t += step_ms
    return wins, ctrs


def _obj_pop_in_window(psth_obj, t_ms_arr, win_ms):
    """(cell, obj, time) → (n_obj, n_cell), sum firing in [t_start, t_end)."""
    lo, hi = win_ms
    msk = (t_ms_arr >= lo) & (t_ms_arr < hi)
    return psth_obj[:, :, msk].sum(-1).T


def _loo_cross_epoch_decode(X_train, X_test, y):
    """LOO across objects; train and test may come from different time windows."""
    n_obj = len(y)
    preds = np.empty(n_obj, dtype=int)
    for i in range(n_obj):
        tr = np.ones(n_obj, dtype=bool)
        tr[i] = False
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))
        clf.fit(X_train[tr], y[tr])
        preds[i] = clf.predict(X_test[i:i + 1])[0]
    return float((preds == y).mean())


_win_defs, _win_ctrs = _time_windows(_cross_t_lo, _cross_t_hi, _cross_win_ms, _cross_step_ms)
_n_win = len(_win_defs)
X_by_win = [_obj_pop_in_window(bfct_avg, t_ms_full, w) for w in _win_defs]

# cross_acc[train, test]; heatmap: y=train, x=test, diagonal=LOO
cross_acc = np.zeros((_n_win, _n_win))
for i_tr in range(_n_win):
    for i_te in range(_n_win):
        cross_acc[i_tr, i_te] = _loo_cross_epoch_decode(
            X_by_win[i_tr], X_by_win[i_te], y_cat)

win_lbl = [f'{c}' for c in _win_ctrs]
cross_df = pd.DataFrame(cross_acc, index=win_lbl, columns=win_lbl)
cross_df.index.name = 'Train (ms)'
cross_df.columns.name = 'Test (ms)'
print(cross_df.round(3).to_string())
print(f'LOO accuracy mean (diag): {np.diag(cross_acc).mean():.3f}')
print(f'Cross-time mean (off-diag): {cross_acc[~np.eye(_n_win, dtype=bool)].mean():.3f}')
#%%
vmax = 0.6
_tick_ms = np.arange(200, 2001, 200)
_tick_pos = np.argmin(np.abs(np.array(_win_ctrs)[:, None] - _tick_ms), axis=0) + 0.5
_tick_lbl = [f'{t}' for t in _tick_ms]

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(cross_acc, vmin=_chance, cmap='Reds',
            square=True, xticklabels=False, yticklabels=False,
            vmax=vmax, ax=ax)
ax.set_xticks(_tick_pos)
ax.set_xticklabels(_tick_lbl)
ax.set_yticks(_tick_pos)
ax.set_yticklabels(_tick_lbl)
ax.plot([0, _n_win], [0, _n_win], color='gray', ls='--', lw=1.2, zorder=10, clip_on=False)
ax.set_xlabel('Test window (ms)')
ax.set_ylabel('Train window (ms)')
ax.set_title('Cross-time category decoding — ML neurons')
fig.tight_layout()
plt.show()

#%% 1) Within-time LOO accuracy vs window position
loo_by_win = np.diag(cross_acc)
win_ctrs_arr = np.array(_win_ctrs)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(win_ctrs_arr, loo_by_win, 'o-', lw=2, color='#d62728', ms=5)
ax.axhline(_chance, color='gray', ls='--', lw=1, label=f'chance ({_chance:.2f})')
ax.set_xlim(_cross_t_lo, _cross_t_hi)
ax.set_ylim(0, max(0.55, loo_by_win.max() + 0.05))
ax.set_xlabel('Time window center (ms)')
ax.set_ylabel('LOO decode accuracy')
ax.set_title('Within-epoch decoding — ML neurons')
ax.legend(frameon=False, loc='best')
fig.tight_layout()
plt.show()

#%% 2) Matrix asymmetry by train–test lag
# lag = test − train (ms); asymmetry = acc(train,test) − acc(test,train)
lag_rows = []
for i_tr in range(_n_win):
    for i_te in range(_n_win):
        if i_tr == i_te:
            continue
        lag_rows.append({
            'lag_ms': win_ctrs_arr[i_te] - win_ctrs_arr[i_tr],
            'asym': cross_acc[i_tr, i_te] - cross_acc[i_te, i_tr],
        })
lag_summary = (pd.DataFrame(lag_rows)
               .groupby('lag_ms', as_index=False)
               .agg(asym_mean=('asym', 'mean'))
               .sort_values('lag_ms'))
asym_plot = lag_summary.iloc[1:-1]   # drop extreme boundary lags

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(asym_plot['lag_ms'], asym_plot['asym_mean'], 'o-', lw=2, color='#4c72b0', ms=4)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('Test − Train lag (ms)')
# ax.set_ylabel('Mean asymmetry (acc$_{tr→te}$ − acc$_{te→tr}$)')
ax.set_ylabel('Correct Rate Difference')
ax.set_title('Matrix asymmetry by lag')
fig.tight_layout()
plt.show()

#%%
