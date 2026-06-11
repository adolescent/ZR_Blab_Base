
#%%
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
import warnings
import numpy as np
warnings.filterwarnings("ignore")

data_path = r'E:\#Preprocessed_Data\Selected_Cells\Ani_Texform\MSB'
# data_path = r'E:\#Preprocessed_Data\Selected_Cells\Ani_Texform\ML'
cell_tag = 'MSB' if data_path.rstrip('\\/').endswith('MSB') else 'ML'

texform_z = np.load(ot.Join(data_path, 'avr_rsp_z.npy'), allow_pickle=True)
fob_rsp = np.load(ot.Join(data_path, 'fob_rsp.npy'), allow_pickle=True)
avr_psth = np.load(ot.Join(data_path, 'avr_psth.npy'), allow_pickle=True)
n_cell = texform_z.shape[0]

# orig per 120-block: AniB | ObjB | AniS | ObjS  →  plot: AniB | AniS | ObjB | ObjS
_blk4 = lambda s: np.r_[s + np.arange(30), s + 60 + np.arange(30),
                        s + 30 + np.arange(30), s + 90 + np.arange(30)]
plot_idx = np.concatenate([_blk4(s) for s in (0, 120, 240, 360)])
texform_hm = texform_z[:, plot_idx]

_blk_starts = (0, 120, 240, 360)
_blk_lbl = ['Low Real', 'Low Texform', 'High Real', 'High Texform']
_sub_lbl = ['AB', 'AS', 'OB', 'OS']

#%% FOB heatmap
fob_rsp_z = (fob_rsp - fob_rsp.mean(1, keepdims=True)) / fob_rsp.std(1, keepdims=True)
fig, ax = plt.subplots(figsize=(3, 6))
sns.heatmap(fob_rsp_z, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar=False, ax=ax)
n_fob = fob_rsp_z.shape[1]
for x in (n_fob // 3, 2 * n_fob // 3):
    ax.axvline(x, color='yellow', lw=2)
ax.set_title('FOB Response')
ax.set_xlabel('Body    |    Face    |  Object')
ax.set_ylabel(f'N_Cell={n_cell}')
fig.tight_layout()
plt.show()

#%% Texform redplot: 4×(AniB–AniS–ObjB–ObjS), blocks = LowReal | LowTex | HighReal | HighTex
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(texform_hm, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar=False, ax=ax)
ax.axvline(240, color='k', lw=2)                         # LowCon | HighCon
for x in (120, 360):                                     # Real | Texform
    ax.axvline(x, color='C0', lw=2)
for b in _blk_starts:
    for x in (b + 30, b + 60, b + 90):                   # AB | AS | OB | OS
        ax.axvline(x, color='yellow', lw=1)
for bi, b in enumerate(_blk_starts):
    ax.text(b + 60, 1.05, _blk_lbl[bi], transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    for si, sl in enumerate(_sub_lbl):
        ax.text(b + 15 + si * 30, 1.01, sl, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7)
ax.set_xlabel('LowCon Real  |  LowCon Texform  |  HighCon Real  |  HighCon Texform')
ax.set_ylabel(f'N_Cell={n_cell}')
# ax.set_title(f'Red plot of {cell_tag} neurons, z-scored')
fig.tight_layout()
plt.show()

#%% RSA (same order as redplot above)
rsa = np.corrcoef(texform_hm.T)
np.fill_diagonal(rsa, 0)

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(rsa, vmin=-0.4, vmax=0.4, center=0, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, square=True, cbar=False, ax=ax)
ax.axvline(240, color='k', lw=2)
ax.axhline(240, color='k', lw=2)
for x in (120, 360):
    ax.axvline(x, color='C0', lw=2)
    ax.axhline(x, color='C0', lw=2)
for b in _blk_starts:
    for x in (b + 30, b + 60, b + 90):
        ax.axvline(x, color='yellow', lw=1)
        ax.axhline(x, color='yellow', lw=1)
for bi, b in enumerate(_blk_starts):
    ax.text(b + 60, 1.07, _blk_lbl[bi], transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(-0.06, b + 60, _blk_lbl[bi], transform=ax.get_yaxis_transform(),
            ha='right', va='center', rotation=90, fontsize=9, fontweight='bold')
    for si, sl in enumerate(_sub_lbl):
        yc = b + 15 + si * 30
        ax.text(yc, 1.02, sl, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7)
        ax.text(-0.02, yc, sl, transform=ax.get_yaxis_transform(),
                ha='right', va='center', rotation=90, fontsize=7)
# ax.set_title(f'RSA of {cell_tag} neurons (Pearson r)')
fig.tight_layout()
plt.show()

#%% Subtype PSTH: Low/High × Real/Texform (population average over selected objs)
# subtype: Ani_Big | Obj_Big | Ani_Small | Obj_Small
# obj_range: single int, slice, or list — indices 0–29 within subtype
subtype = 'Ani_Big'
obj_range = slice(None)    # all 30; e.g. 5, slice(0, 10), [0, 3, 7]

_subtype_off = {'Ani_Big': 0, 'Obj_Big': 30, 'Ani_Small': 60, 'Obj_Small': 90}
_blk_off = {'LowCon Real': 0, 'LowCon Texform': 120, 'HighCon Real': 240, 'HighCon Texform': 360}
_t_plot_lo, _t_plot_hi = -100, 350
bin_ms = 5
_t_win = slice(150, 320)  # redplot window → 50..220 ms on t_ms axis


def _normalize_obj_range(rng):
    if isinstance(rng, int):
        idx = np.array([rng], dtype=int)
    elif isinstance(rng, slice):
        idx = np.arange(30)[rng]
    else:
        idx = np.asarray(rng, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError('obj_range must select at least one object (0–29)')
    if (idx < 0).any() or (idx >= 30).any():
        raise ValueError(f'obj indices out of range 0–29: {idx.tolist()}')
    return idx


def _obj_range_label(idx):
    if idx.size == 1:
        return f'obj {idx[0] + 1}'
    if idx.size == 30:
        return 'all 30 objs'
    if np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)):
        return f'objs {idx[0] + 1}–{idx[-1] + 1}'
    return 'objs ' + ', '.join(str(i + 1) for i in idx)


def _avg_cond_fr(psth, blk_start, sub_start, obj_idx, time_msk, n_t, bin_ms):
    cols = blk_start + sub_start + obj_idx
    dat = psth[:, cols, :][:, :, time_msk][..., :n_t].mean(1)  # avg over objs
    return dat.reshape(n_cell, -1, bin_ms).mean(-1) * 1000


obj_idx = _normalize_obj_range(obj_range)
sub_off = _subtype_off[subtype]
t_ms = np.arange(-100, -100 + avr_psth.shape[-1])
msk = (t_ms >= _t_plot_lo) & (t_ms <= _t_plot_hi)
n_t = int(msk.sum()) // bin_ms * bin_ms
t_plot = t_ms[msk][:n_t].reshape(-1, bin_ms).mean(1)

_cond_specs = [
    ('LowCon Real', _blk_off['LowCon Real']),
    ('LowCon Texform', _blk_off['LowCon Texform']),
    ('HighCon Real', _blk_off['HighCon Real']),
    ('HighCon Texform', _blk_off['HighCon Texform']),
]
# Real = green, Texform = red; lighter = LowCon, darker = HighCon
_cond_colors = ['#98df8a', '#ff9896', '#2ca02c', '#d62728']

fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(t_ms[_t_win.start], t_ms[_t_win.stop - 1], color='lightgreen', alpha=0.2, zorder=0)
ax.axvline(0, color='gray', ls='--', lw=1)
for (lbl, blk), c in zip(_cond_specs, _cond_colors):
    fr = _avg_cond_fr(avr_psth, blk, sub_off, obj_idx, msk, n_t, bin_ms)
    y = fr.mean(0)
    err = fr.std(0, ddof=1) / np.sqrt(n_cell)
    ax.plot(t_plot, y, lw=2, color=c, label=lbl)
    ax.fill_between(t_plot, y - err, y + err, color=c, alpha=0.2, linewidth=0)
ax.set_xlim(_t_plot_lo, _t_plot_hi)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate (Hz)')
ax.set_title(f'{subtype} {_obj_range_label(obj_idx)} — {cell_tag} neurons (N={n_cell})')
ax.legend(frameon=False, loc='best')
fig.tight_layout()
plt.show()

#%%

