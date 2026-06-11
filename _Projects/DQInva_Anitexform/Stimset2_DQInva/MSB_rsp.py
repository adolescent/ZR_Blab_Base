

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
data_path = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable\MSB'
# data_path = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable\ML'
savepath = ot.Join(data_path,'Set2')
set2_stims = np.load(ot.Join(data_path,'set2_rsp_z.npy'),allow_pickle=True)
fob_rsp = np.load(ot.Join(data_path,'fob_rsp.npy'),allow_pickle=True)
psth_avr = np.load(ot.Join(data_path,'set2_psth.npy'),allow_pickle=True)

#%% FOB heatmap
fob_rsp_z = (fob_rsp-fob_rsp.mean(1,keepdims=True))/fob_rsp.std(1,keepdims=True)
fig, ax = plt.subplots(figsize=(3, 6))
sns.heatmap(fob_rsp_z, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False,cbar=False,
             ax=ax,)
n_fob = fob_rsp_z.shape[1]
for x in (n_fob // 3, 2 * n_fob // 3):
    ax.axvline(x, color='yellow', lw=2)
n_cell = fob_rsp_z.shape[0]
ax.set_title('FOB Response')
ax.set_xlabel('Body    |    Face    |  Object')
ax.set_ylabel(f'N_Cell={n_cell}')
fig.tight_layout()
plt.show()

#%% Body PSTH: avg 20 objs × 4 cond (Tex_CTR, Shading_CTR, Tex, Shading)
# set2 order: 80×(body|face|fruit); per 80: 20obj×[Tex_CTR, Shading_CTR, Tex, Shading]
bin_ms = 5
t_ms = np.arange(-100, -100 + psth_avr.shape[-1])          # 1 ms bins, -100..349 ms
msk = (t_ms >= -100) & (t_ms <= 350)

n_cell = psth_avr.shape[0]
body_psth = psth_avr[:, :80, :][:, :, msk].reshape(n_cell, 20, 4, -1)
body_cond = body_psth.mean(1)                               # avg over 20 body images

n_t = body_cond.shape[-1] // bin_ms * bin_ms
t_plot = t_ms[msk][:n_t].reshape(-1, bin_ms).mean(1)
body_fr = body_cond[..., :n_t].reshape(n_cell, 4, -1, bin_ms).mean(-1) * 1000  # Hz

# orig cond dim: [Tex_CTR, Shading_CTR, Texture, Shading] → plot: Shading, Shading_CTR, Texture, Tex_CTR
cond_idx = [3, 1, 2, 0]
cond_lbl = ['Shading', 'Shading_CTR', 'Texture', 'Tex_CTR']
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(50, 320, color='lightgreen', alpha=0.2, zorder=0)  # t_win 150:420 → 50..320 ms
ax.axvline(0, color='gray', ls='--', lw=1)
for lbl, c, ci in zip(cond_lbl, colors, cond_idx):
    curves = body_fr[:, ci, :]
    y = curves.mean(0)
    err = curves.std(0, ddof=1) / np.sqrt(curves.shape[0])
    ax.plot(t_plot, y, lw=2, color=c, label=lbl)
    ax.fill_between(t_plot, y - err, y + err, color=c, alpha=0.2, linewidth=0)
ax.set_xlim(-100, 350)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate (Hz)')
ax.set_title(f'Body response of MSB neurons')
# ax.set_title(f'Fruit response of ML neurons')
ax.legend(frameon=False, loc='best')
fig.tight_layout()
plt.show()

#%% Body PSTH: single / selected neurons (same 4 cond, 5 ms bins)
# cell_idx: 0-based index into pooled cells — int, list, slice, or ndarray
# cell_idx = np.arange(30,50)              # e.g. 3, [0, 2, 5], slice(0, 5), np.arange(10)
# cell_idx = np.arange(200,201)
cell_idx = 190
bin_ms = 5


def _normalize_cell_idx(rng, n):
    if isinstance(rng, int):
        idx = np.array([rng], dtype=int)
    elif isinstance(rng, slice):
        idx = np.arange(n)[rng]
    else:
        idx = np.asarray(rng, dtype=int).ravel()
    if idx.size == 0:
        raise ValueError('cell_idx must select at least one neuron')
    if (idx < 0).any() or (idx >= n).any():
        raise ValueError(f'cell indices out of range 0–{n - 1}: {idx.tolist()}')
    return idx


def _cell_idx_label(idx, n):
    if idx.size == 1:
        return f'cell {idx[0]}'
    if idx.size == n:
        return f'all {n} cells'
    if np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
        return f'cells {idx[0]}–{idx[-1]}'
    return 'cells ' + ', '.join(str(i) for i in idx)


_n_cell = psth_avr.shape[0]
cells = _normalize_cell_idx(cell_idx, _n_cell)
_t_ms = np.arange(-100, -100 + psth_avr.shape[-1])
_msk = (_t_ms >= -100) & (_t_ms <= 350)
_sel_psth = psth_avr[cells, :80, :][:, :, _msk].reshape(cells.size, 20, 4, -1)
_sel_cond = _sel_psth.mean(1)                                    # avg over 20 body images
_n_t = _sel_cond.shape[-1] // bin_ms * bin_ms
_t_plot = _t_ms[_msk][:_n_t].reshape(-1, bin_ms).mean(1)
fr_sel = (_sel_cond[..., :_n_t]
          .reshape(cells.size, 4, -1, bin_ms).mean(-1) * 1000)   # Hz

_cond_idx = [3, 1, 2, 0]
_cond_lbl = ['Shading', 'Shading_CTR', 'Texture', 'Tex_CTR']
_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(50, 320, color='lightgreen', alpha=0.2, zorder=0)
ax.axvline(0, color='gray', ls='--', lw=1)
for lbl, c, ci in zip(_cond_lbl, _colors, _cond_idx):
    curves = fr_sel[:, ci, :]
    y = curves.mean(0)
    ax.plot(_t_plot, y, lw=2, color=c, label=lbl)
    if curves.shape[0] > 1:
        err = curves.std(0, ddof=1) / np.sqrt(curves.shape[0])
        ax.fill_between(_t_plot, y - err, y + err, color=c, alpha=0.2, linewidth=0)
ax.set_xlim(-100, 350)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate (Hz)')
ax.set_title(f'Body response — {_cell_idx_label(cells, _n_cell)} (N={cells.size}, {bin_ms} ms bin)')
ax.legend(frameon=False, loc='best')
fig.tight_layout()
plt.show()

#%% Body Shading binned raster: all neurons (optional range), avg 20 objs, 10 ms bin
# cell_range: int, list, slice, or None for all — e.g. slice(None), slice(0, 50), [0, 5, 10]
cell_range = slice(None)
raster_bin_ms = 5


def _pop_cell_idx(rng, n):
    if rng is None:
        return np.arange(n, dtype=int)
    if isinstance(rng, int):
        return np.array([rng], dtype=int)
    if isinstance(rng, slice):
        return np.arange(n)[rng]
    idx = np.asarray(rng, dtype=int).ravel()
    if idx.size == 0:
        raise ValueError('cell_range must select at least one neuron')
    if (idx < 0).any() or (idx >= n).any():
        raise ValueError(f'cell indices out of range 0–{n - 1}: {idx.tolist()}')
    return idx


def _pop_cell_label(idx, n):
    if idx.size == 1:
        return f'cell {idx[0]}'
    if idx.size == n:
        return f'all {n} cells'
    if np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
        return f'cells {idx[0]}–{idx[-1]}'
    return f'{idx.size} cells'


_n_pop = psth_avr.shape[0]
pop_cells = _pop_cell_idx(cell_range, _n_pop)

_t_pop = np.arange(-100, -100 + psth_avr.shape[-1])
_msk_pop = (_t_pop >= -100) & (_t_pop <= 350)
_pop_psth = psth_avr[pop_cells, :80, :][:, :, _msk_pop].reshape(pop_cells.size, 20, 4, -1)
_sh_k = 3  # real Shading in [Tex_CTR, Shading_CTR, Tex, Shading]
_sh_pop = _pop_psth[:, :, _sh_k, :].mean(1)                      # avg 20 body objs
_n_tr = _sh_pop.shape[-1] // raster_bin_ms * raster_bin_ms
_t_rast = _t_pop[_msk_pop][:_n_tr].reshape(-1, raster_bin_ms).mean(1)
raster_pop = (_sh_pop[..., :_n_tr]
              .reshape(pop_cells.size, -1, raster_bin_ms).mean(-1) * 1000)
raster_norm = raster_pop / np.maximum(raster_pop.max(1, keepdims=True), 1e-6)  # peak = 1 per neuron

from matplotlib.patches import Rectangle

_highlight_cells = [190, 300]  # neuron indices to frame in the raster

fig, ax = plt.subplots(figsize=(4, 7))
im = ax.imshow(
    raster_norm, aspect='auto', origin='lower', interpolation='nearest',
    extent=(_t_rast[0], _t_rast[-1] + raster_bin_ms, -0.5, pop_cells.size - 0.5),
    cmap='Greys', vmin=0, vmax=1,
)
ax.axvspan(50, 320, color='lime', alpha=0.15, zorder=0)
ax.axvline(0, color='cyan', ls='--', lw=1)
ax.set_xlim(-100, 350)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron index')
_ytick_cells = np.arange(0, int(pop_cells.max()) + 1, 50)
_ytick_pos = np.searchsorted(pop_cells, _ytick_cells)
_valid = (_ytick_pos < pop_cells.size) & (pop_cells[_ytick_pos] == _ytick_cells)
ax.set_yticks(_ytick_pos[_valid])
ax.set_yticklabels(_ytick_cells[_valid], fontsize=8)
_x0, _x1 = ax.get_xlim()
for _cid in _highlight_cells:
    _hit = np.where(pop_cells == _cid)[0]
    if _hit.size == 0:
        continue
    _row = int(_hit[0])
    ax.add_patch(Rectangle(
        (_x0, _row - 0.5), _x1 - _x0, 1.0,
        fill=False, edgecolor='yellow',alpha=0.7, lw=1, zorder=5, clip_on=False,
    ))
# fig.colorbar(im, ax=ax, label='Peak-normalized response', shrink=0.85, pad=0.02)
# ax.set_title(f'Body Shading raster — {_pop_cell_label(pop_cells, _n_pop)} ' f'(avg 20 objs, {raster_bin_ms} ms bin)')
fig.tight_layout()
plt.show()

#%% Set2 heatmap: 3×(Body|Face|Fruit), each 4×20 = Sh–Tx–ShC–TxC
# orig per category: 20obj×[Tex_CTR, Shading_CTR, Texture, Shading]
cond_ord = [3, 2, 1, 0]  # → Shading, Texture, Shading_CTR, Tex_CTR
idx = [b + r * 4 + c for b in (0, 80, 160) for c in cond_ord for r in range(20)]
set2_hm = set2_stims[:, idx]
n_cell = set2_hm.shape[0]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(set2_hm, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar=False, ax=ax)
for x in (80, 160):                       # Body | Face | Fruit
    ax.axvline(x, color='yellow', lw=2)
for b in (0, 80, 160):
    for x in (b + 20, b + 40, b + 60):    # Sh | Tx | ShC | TxC within each group
        ax.axvline(x, color='k', lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel('Body | Face | Fruit; each: 20 Sh – 20 Tx – 20 ShC – 20 TxC')
ax.set_ylabel(f'N_Cell={n_cell}')
# ax.set_title('Red plot of MSB neurons, z-scored')
ax.set_title('Red plot of ML neurons, z-scored')
fig.tight_layout()
plt.show()

#%% RSA (same order as redplot above)
rsa = np.corrcoef(set2_hm.T)
np.fill_diagonal(rsa, 0)

_sub_lbl = ['B', 'F', 'Fr']
_cond_lbl = ['Sh', 'Tx', 'ShC', 'TxC']

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(rsa, vmin=-0.5, vmax=0.5, center=0, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, square=True, cbar=False, ax=ax)
for x in (80, 160):
    ax.axvline(x, color='yellow', lw=2)
    ax.axhline(x, color='yellow', lw=2)
for b in (0, 80, 160):
    for x in (b + 20, b + 40, b + 60):
        ax.axvline(x, color='k', lw=0.8, ls='--', alpha=0.6)
        ax.axhline(x, color='k', lw=0.8, ls='--', alpha=0.6)
for i, sl in enumerate(_sub_lbl):
    ax.text(40 + i * 80, 1.06, sl, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    for j, cl in enumerate(_cond_lbl):
        ax.text(10 + i * 80 + j * 20, 1.01, cl, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7)
# ax.set_title('RSA of MSB neurons (Pearson r)')
fig.tight_layout()
plt.show()

#%% Stimulus correlation: Sh–Tx, Sh–ShC, Tx–TxC (Body / Face / Fruit)
from scipy import stats
from matplotlib.patches import Patch

R = set2_stims
_corr_pairs = [('Sh_Tx', 3, 2), ('Sh_ShC', 3, 1), ('Tx_TxC', 2, 0)]
rows = []
for sub, s in zip(['Body', 'Face', 'Fruit'], [0, 80, 160]):
    for r in range(20):
        row = {'Subclass': sub}
        for name, c1, c2 in _corr_pairs:
            i1, i2 = s + r * 4 + c1, s + r * 4 + c2
            row[name] = np.corrcoef(R[:, i1], R[:, i2])[0, 1]
        rows.append(row)
corr_df = pd.DataFrame(rows)

cols = [p[0] for p in _corr_pairs]
lbls = ['Shading–Texture', 'Shading–Shading_CTR', 'Texture–Texture_CTR']
cols_c = ['#c44e52', '#dd8452', '#55a868']
test_pairs = [(0, 1), (0, 2), (1, 2)]


def _pstar(p):
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _bracket(ax, x1, x2, y, s, dh=0.022):
    ax.plot([x1, x1, x2, x2], [y, y + dh, y + dh, y], color='#333', lw=0.9, clip_on=False)
    ax.text((x1 + x2) / 2, y + dh, s, ha='center', va='bottom', fontsize=8, clip_on=False)


fig, ax = plt.subplots(figsize=(5, 6))
x0, w, step = np.arange(3) * 3.0, 0.48, 0.62
y_top = -np.inf
for gi, sub in enumerate(['Body', 'Face', 'Fruit']):
    d = corr_df.query('Subclass == @sub')
    x = x0[gi] + np.arange(3) * step
    m, e = d[cols].mean().values, d[cols].sem().values
    ax.bar(x, m, w, yerr=e, capsize=2.5, color=cols_c,
           edgecolor='white', linewidth=0.8, zorder=3, error_kw={'lw': 0.9})
    for _, row in d.iterrows():
        ax.plot(x[:2], row[cols[:2]].values, color='gray', lw=0.7, alpha=0.3, zorder=2)
    y0 = (m + e).max() + 0.035
    for k, (a, b) in enumerate(test_pairs):
        p = stats.ttest_rel(d[cols[a]], d[cols[b]]).pvalue
        _bracket(ax, x[a], x[b], y0 + k * 0.065, _pstar(p))
    y_top = max(y_top, y0 + (len(test_pairs) - 1) * 0.065 + 0.05)

ax.set_xticks(x0 + step)
ax.set_xticklabels(['Body', 'Face', 'Fruit'])
ax.set_ylabel('Pearson r')
ax.spines[['top', 'right']].set_visible(False)
# ax.set_ylim(top=max(y_top, ax.get_ylim()[1]))
ax.set_ylim(bottom=-0.2, top=0.8)
ax.legend(handles=[Patch(facecolor=c, edgecolor='white', label=l) for c, l in zip(cols_c, lbls)],
          frameon=False, loc='upper right', fontsize=9)
fig.tight_layout()
plt.show()

#%% Set2 cross-decoding (LOO linear SVM, trial-level data from sites)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DQInva_site_path = r'E:\#Preprocessed_Data\SiteClass\DQInva'
_ceiling, _dp_thres = 0.2, 0.5
_t_win = slice(150, 420)
_n_splits = 14  # 14 repeats per stimulus for LOO


def _pair_avg(x):
    """(cell, trial, 2*n_img) -> (cell, trial, n_img), average duplicate stim blocks."""
    return x.reshape(x.shape[0], x.shape[1], 2, -1).mean(2)


def _load_trial_rsp(prefer):
    """Load trial-level Set2 responses from all sites and pool MSB/ML cells.

    Returns
    -------
    rsp : np.ndarray
        (n_cell, 14, 240) — columns aligned to stim_info rows 360:600 after
        Tsv_Txt_Align, i.e. obj1[Tex_CTR, Shading_CTR, Tex, Shading], obj2[...],
        ... through 60 objects (Body 1–20, Face 21–40, Fruit 41–60).
    set2_info : pd.DataFrame
        stim_info rows 360:600 (240 stims), one row per rsp column.
    """
    site_paths = ot.Get_File_Name(DQInva_site_path, '.joblib')
    all_rsp = []
    set2_info = None
    for sp in site_paths:
        a = JL.load(sp)
        if set2_info is None:
            set2_info = a.stim_info.iloc[360:600].reset_index(drop=True)
        cells, _ = a.Cell_Selection(ceiling=_ceiling, prefer=prefer, dp_thres=_dp_thres)
        trial_rsp = a.raw_psth[cells][:, :, 360:, _t_win].sum(-1)  # (cell, trial, 480)
        rsp = _pair_avg(trial_rsp)  # (cell, trial, 240)
        assert rsp.shape[2] == 240
        assert rsp.shape[1] == _n_splits
        all_rsp.append(rsp)
    return np.vstack(all_rsp), set2_info


def _subclass_of_object(object_id):
    if 1 <= object_id <= 20:
        return 'Body'
    if 21 <= object_id <= 40:
        return 'Face'
    if 41 <= object_id <= 60:
        return 'Fruit'
    raise ValueError(f'object_id out of range: {object_id}')


def _col_from_info(info, object_id, cond):
    """Map (Object_ID, condition) -> rsp column index via stim_info metadata."""
    sub = _subclass_of_object(object_id)
    cat = f'{sub}_{cond}'
    hit = info.index[(info.Category == cat) & (info.Object.astype(int) == object_id)]
    if len(hit) != 1:
        raise ValueError(f'ambiguous/missing stim: Object={object_id}, cond={cond}')
    return int(hit[0])


def _cols_for_objects(info, object_ids, cond):
    """One column per object; list order defines SVM class label 0..N-1."""
    return [_col_from_info(info, oid, cond) for oid in object_ids]


def _build_label_map(info):
    """Audit table: SVM label -> Object_ID -> rsp column -> Category."""
    rows = []
    task_specs = [
        ('Body', list(range(1, 21))),
        ('Face', list(range(21, 41))),
        ('Fruit', list(range(41, 61))),
        ('All', list(range(1, 61))),
    ]
    for task, obj_ids in task_specs:
        for label, oid in enumerate(obj_ids):
            for cond in CONDS:
                col = _col_from_info(info, oid, cond)
                rows.append({
                    'Task': task,
                    'Label': label,
                    'Object_ID': int(oid),
                    'Condition': cond,
                    'Column': col,
                    'Category': info.loc[col, 'Category'],
                    'FileName': info.loc[col, 'FileName'],
                })
    return pd.DataFrame(rows)


def _verify_column_layout(info):
    """Confirm rsp columns follow obj×[Tex_CTR, Shading_CTR, Tex, Shading]."""
    for oid in range(1, 61):
        cols = [_col_from_info(info, oid, c) for c in CONDS]
        assert cols == sorted(cols), f'non-contiguous block for object {oid}'
        if oid < 60:
            next_ctr = _col_from_info(info, oid + 1, 'Tex_CTR')
            assert cols[-1] + 1 == next_ctr, f'column gap after object {oid}'
    # spot-check first object matches TSV row 360
    assert info.loc[0, 'Category'] == 'Body_Tex_CTR' and int(info.loc[0, 'Object']) == 1


prefer = JL.load(ot.Join(data_path, 'pseudo_trials.joblib'))['prefer']
cell_tag = 'MSB' if 'MSB' in data_path.upper() else 'ML'

# trial-level pooled response (n_cell, 14, 240), columns = stim_info 360:600
rsp, set2_info = _load_trial_rsp(prefer)
_n_cell = rsp.shape[0]
print(f'Set2 trial-level rsp loaded: {rsp.shape}, prefer={prefer}, pool={cell_tag}')

#%%
# Decoding labels = object identity (Label 0 -> first object in task).
# Columns are NOT the ord4 visualization reorder; they follow raw stim_info order:
# obj1[Tex_CTR, Shading_CTR, Tex, Shading], obj2[...], ... 60 objects total.

SUBCATS = ['Body', 'Face', 'Fruit', 'All']
OBJECT_IDS = {
    'Body': list(range(1, 21)),
    'Face': list(range(21, 41)),
    'Fruit': list(range(41, 61)),
    'All': list(range(1, 61)),
}
CONDS = ['Tex_CTR', 'Shading_CTR', 'Tex', 'Shading']

_verify_column_layout(set2_info)
label_map_df = _build_label_map(set2_info)
print('Label map (first 8 rows):')
print(label_map_df.head(8).to_string(index=False))
print('Cross-condition label check (Body obj1, Shading vs Tex):')
print(label_map_df.query("Task=='Body' and Object_ID==1 and Condition in ['Shading','Tex']")[
    ['Label', 'Object_ID', 'Condition', 'Column', 'Category']].to_string(index=False))


def _loo_decode(rsp, train_cols, test_cols, n_class):
    """LOO across 14 repeats; return fold-wise accuracies and their mean.

    Parameters
    ----------
    rsp : np.ndarray
        (n_cell, 14, n_stim) trial-level responses.
    train_cols, test_cols : list[int]
        Length-n_class lists with column indices for each class.
    n_class : int
        Number of classes (20 or 60).
    """
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    n_trial = rsp.shape[1]
    accs = []
    for te in range(n_trial):
        X_tr, y_tr = [], []
        for ti in range(n_trial):
            if ti == te:
                continue
            for c in range(n_class):
                X_tr.append(rsp[:, ti, train_cols[c]])
                y_tr.append(c)
        X_tr = np.vstack(X_tr)
        y_tr = np.asarray(y_tr, dtype=int)
        X_te = np.vstack([rsp[:, te, test_cols[c]] for c in range(n_class)])
        y_te = np.arange(n_class, dtype=int)
        clf.fit(X_tr, y_tr)
        accs.append((clf.predict(X_te) == y_te).mean())
    accs = np.asarray(accs, dtype=float)
    return accs, float(accs.mean()), float(accs.std(ddof=1) if len(accs) > 1 else 0.0)


rows = []
chance_20, chance_60 = 1 / 20, 1 / 60

# 20-class tasks: Body / Face / Fruit
for sub in SUBCATS[:3]:
    obj_ids = OBJECT_IDS[sub]
    n_class = len(obj_ids)
    cols_by_cond = {cond: _cols_for_objects(set2_info, obj_ids, cond) for cond in CONDS}
    for tr_cond in CONDS:
        for te_cond in CONDS:
            fold_acc, acc, acc_std = _loo_decode(
                rsp, cols_by_cond[tr_cond], cols_by_cond[te_cond], n_class)
            mode = 'LOO_same_condition' if tr_cond == te_cond else 'LOO_cross_condition'
            rows.append({
                'Cell_pool': cell_tag,
                'Subclass': sub,
                'N_class': n_class,
                'Train_cond': tr_cond,
                'Test_cond': te_cond,
                'Accuracy': acc,
                'CV_Accuracy_Mean': acc,
                'CV_Accuracy_STD': acc_std,
                'CV_N_Fold': len(fold_acc),
                'Eval_mode': mode,
                'Chance': chance_20,
                'N_cell': _n_cell,
            })

# 60-class task: All objects pooled (Label 0..59 -> Object_ID 1..60)
obj_ids_all = OBJECT_IDS['All']
n_class_all = len(obj_ids_all)
cols_by_cond_all = {cond: _cols_for_objects(set2_info, obj_ids_all, cond) for cond in CONDS}
for tr_cond in CONDS:
    for te_cond in CONDS:
        fold_acc, acc, acc_std = _loo_decode(
            rsp, cols_by_cond_all[tr_cond], cols_by_cond_all[te_cond], n_class_all)
        mode = 'LOO_same_condition' if tr_cond == te_cond else 'LOO_cross_condition'
        rows.append({
            'Cell_pool': cell_tag,
            'Subclass': 'All',
            'N_class': n_class_all,
            'Train_cond': tr_cond,
            'Test_cond': te_cond,
            'Accuracy': acc,
            'CV_Accuracy_Mean': acc,
            'CV_Accuracy_STD': acc_std,
            'CV_N_Fold': len(fold_acc),
            'Eval_mode': mode,
            'Chance': chance_60,
            'N_cell': _n_cell,
        })

decode_df = pd.DataFrame(rows)
print(decode_df.head())
#%%
# pivot for visualization
decode_pivot = {
    sub: (decode_df.loc[decode_df['Subclass'] == sub]
          .pivot(index='Train_cond', columns='Test_cond', values='Accuracy')
          .reindex(index=CONDS, columns=CONDS))
    for sub in SUBCATS
}

decode_pivot_mean = (decode_df
                     .loc[decode_df['Subclass'].isin(SUBCATS[:3])]
                     .pivot_table(index='Train_cond', columns='Test_cond',
                                  values='Accuracy', aggfunc='mean')
                     .reindex(index=CONDS, columns=CONDS))

# heatmaps: Body / Face / Fruit (3-panel) + All (separate) + mean across subclasses
plot_order = ['Shading', 'Tex', 'Shading_CTR', 'Tex_CTR']
COND_SHORT = ['Sh', 'Tx', 'SC', 'TC']
name_map = dict(zip(CONDS, ['Tex_CTR', 'Shading_CTR', 'Tex', 'Shading']))

vmin, vmax = 0, 0.6


def _plot_decode_heatmap(mat, ax, title, *, cbar=False):
    mat = mat.rename(index=name_map, columns=name_map).reindex(index=plot_order, columns=plot_order)
    kw = dict(
        ax=ax, cmap='RdBu_r', vmin=vmin, vmax=vmax,
        annot=True, fmt='.2f', linewidths=0.6, square=True, center=0.05,
    )
    if cbar:
        kw['cbar_kws'] = {'label': 'Accuracy', 'shrink': 0.85, 'ticks': np.linspace(vmin, vmax, 5)}
    else:
        kw['cbar'] = False
    sns.heatmap(mat, **kw)
    ax.set_title(title)
    ax.set_xlabel('Test')
    ax.set_ylabel('Train')
    ax.set_xticklabels(COND_SHORT, rotation=0)
    ax.set_yticklabels(COND_SHORT, rotation=0)


fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
for ax, sub in zip(axes, SUBCATS[:3]):
    _plot_decode_heatmap(decode_pivot[sub], ax, f'{sub} (diag = LOO)')
fig.suptitle(f'cross-decoding — {cell_tag} neurons (20-class)', y=1.02)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(5.5, 4.8))
_plot_decode_heatmap(decode_pivot['All'], ax, f'All (60-class; diag = LOO)', cbar=True)
# fig.suptitle(f'cross-decoding — {cell_tag} neurons (60-class)', y=1.02)
fig.tight_layout()
plt.show()
