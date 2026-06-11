

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
from matplotlib.transforms import blended_transform_factory
warnings.filterwarnings("ignore")

# msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB','.joblib')
data_path = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable\MSB'
data_path = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable\ML'
savepath = ot.Join(data_path,'Set2')
set1_stims = np.load(ot.Join(data_path,'set1_rsp_z.npy'),allow_pickle=True)
fob_rsp = np.load(ot.Join(data_path,'fob_rsp.npy'),allow_pickle=True)
set1_psth = np.load(ot.Join(data_path,'set1_psth.npy'),allow_pickle=True)

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

#%% Set1 redplot: Shading|Texture → In|Out → R-C-L
# orig col order: per obj × [In-C, In-R, In-L, Out-C, Out-R, Out-L]; 0:66 shading, 66:132 texture
# plot order: [Shading: In-R×11, In-C×11, In-L×11, Out-R×11, Out-C×11, Out-L×11] + [Texture: same]
_n_obj = 11
_io_rc_l_k = (1, 0, 2, 4, 3, 5)  # In-R, In-C, In-L, Out-R, Out-C, Out-L
_reindex = np.array([o * 6 + k for k in _io_rc_l_k for o in range(_n_obj)])
set1_hm = set1_stims[:, np.r_[_reindex, _reindex + 66]]
n_cell = set1_hm.shape[0]

_l1 = (66,)                               # Shading | Texture
_l2 = (33, 99)                            # In | Out
_l3 = (11, 22, 44, 55, 77, 88, 100, 111)  # R | C | L
_rc_l_lbl = ['R', 'C', 'L', 'R', 'C', 'L']


def _draw_set1_layout(ax, *, both=False):
    """Subtle hierarchy grid: L1 solid, L2 medium, L3 light dashed."""
    z = 5
    for xs, color, lw, ls in (
        (_l3, '#cccccc', 0.6, '--'),
        (_l2, '#888888', 1.0, '-'),
        (_l1, '#333333', 1.4, '-'),
    ):
        for x in xs:
            ax.axvline(x, color=color, lw=lw, ls=ls, zorder=z)
            if both:
                ax.axhline(x, color=color, lw=lw, ls=ls, zorder=z)


def _annotate_set1_hierarchy(ax, sides=('top',)):
    """Three label tiers; top/left only to avoid overlap on RSA."""
    if 'top' in sides:
        tx = blended_transform_factory(ax.transData, ax.transAxes)
        for i, lab in enumerate(_rc_l_lbl * 2):
            ax.text(5.5 + i * _n_obj, 1.01, lab, transform=tx,
                    ha='center', va='bottom', fontsize=7, color='#555')
        for off, lab in ((0, 'In'), (33, 'Out'), (66, 'In'), (99, 'Out')):
            ax.text(off + 16.5, 1.05, lab, transform=tx,
                    ha='center', va='bottom', fontsize=8, color='#333')
        for off, lab in ((0, 'Shading'), (66, 'Texture')):
            ax.text(off + 33, 1.10, lab, transform=tx,
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    if 'bottom' in sides:
        bx = blended_transform_factory(ax.transData, ax.transAxes)
        for i, lab in enumerate(_rc_l_lbl * 2):
            ax.text(5.5 + i * _n_obj, -0.01, lab, transform=bx,
                    ha='center', va='top', fontsize=7, color='#555')
        for off, lab in ((0, 'In'), (33, 'Out'), (66, 'In'), (99, 'Out')):
            ax.text(off + 16.5, -0.05, lab, transform=bx,
                    ha='center', va='top', fontsize=8, color='#333')
        for off, lab in ((0, 'Shading'), (66, 'Texture')):
            ax.text(off + 33, -0.10, lab, transform=bx,
                    ha='center', va='top', fontsize=9, fontweight='bold')


fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(set1_hm, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, cbar=False, ax=ax)
_draw_set1_layout(ax)
_annotate_set1_hierarchy(ax, sides=('top',))
ax.set_xlabel('Object (1–11 per R/C/L block)')
ax.set_ylabel(f'N_Cell={n_cell}')
fig.subplots_adjust(top=0.82, bottom=0.08)
plt.show()

#%% Identity PSTH: Shading vs Texture — 2 curves
obj_id = 9       # 1–11
io = 'In'        # 'In' or 'Out'
bin_ms = 5
t_ms = np.arange(-100, -100 + set1_psth.shape[-1])
msk = (t_ms >= -100) & (t_ms <= 350)

# per obj × [In-C, In-R, In-L, Out-C, Out-R, Out-L]; 0:66 shading, 66:132 texture
_base = (obj_id - 1) * 6
_io_k = (0, 1, 2) if io == 'In' else (3, 4, 5)   # C, R, L
sh_cols = [_base + k for k in _io_k]
tx_cols = [_base + 66 + k for k in _io_k]

n_cell = set1_psth.shape[0]
n_t = int(msk.sum()) // bin_ms * bin_ms
t_plot = t_ms[msk][:n_t].reshape(-1, bin_ms).mean(1)


def _psth_fr(cols):
    """Population mean ± SEM (Hz); avg R-C-L per cell, then across neurons."""
    dat = set1_psth[:, cols, :][:, :, msk][..., :n_t]
    dat = dat.reshape(n_cell, len(cols), -1, bin_ms).mean(-1) * 1000
    curves = dat.mean(1)
    return curves.mean(0), curves.std(0, ddof=1) / np.sqrt(n_cell)


fig, ax = plt.subplots(figsize=(6, 4))
ax.axvspan(50, 320, color='lightgreen', alpha=0.2, zorder=0)
ax.axvline(0, color='gray', ls='--', lw=1)
for lbl, cols, c in [('Shading', sh_cols, '#c44e52'), ('Texture', tx_cols, '#4c72b0')]:
    y, err = _psth_fr(cols)
    ax.plot(t_plot, y, lw=2.2, color=c, label=lbl)
    ax.fill_between(t_plot, y - err, y + err, color=c, alpha=0.2, linewidth=0)
ax.set_xlim(-100, 350)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate (Hz)')
ax.set_title(f'Object {obj_id} ({io}) — MSB population PSTH (N={n_cell})')
ax.legend(frameon=False, loc='best')
fig.tight_layout()
plt.show()

#%% RSA (same order as redplot above)
rsa = np.corrcoef(set1_hm.T)
np.fill_diagonal(rsa, 0)

fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(rsa, vmin=-0.5, vmax=0.5, center=0, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, square=True,
            linewidths=0, ax=ax)
_draw_set1_layout(ax, both=True)
_annotate_set1_hierarchy(ax, sides=('top', 'left'))
fig.subplots_adjust(top=0.78, left=0.06, right=0.98, bottom=0.18)
plt.show()

#%% Identity-wise correlations + shuffled-identity control (×5)
rng = np.random.default_rng(20260608)
_n_shuffle = 5

# (cell, obj, stim_type, cond_k); st: 0=Shading, 1=Texture
# k: 0=In-C, 1=In-R, 2=In-L, 3=Out-C, 4=Out-R, 5=Out-L
X = np.empty((n_cell, _n_obj, 2, 6))
X[:, :, 0, :] = set1_stims[:, :66].reshape(n_cell, _n_obj, 6)
X[:, :, 1, :] = set1_stims[:, 66:].reshape(n_cell, _n_obj, 6)

_IN_K = {'R': 1, 'C': 0, 'L': 2}
_OUT_K = {'R': 4, 'C': 3, 'L': 5}
_ori_pairs = [('R', 'C'), ('R', 'L'), ('C', 'L')]


def _safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


_K_LBL = ['In-C', 'In-R', 'In-L', 'Out-C', 'Out-R', 'Out-L']


def _corr_rows(perm=None, *, condition='real', shuffle_id=np.nan):
    """Unit-level correlations; shuffle perm breaks identity on the 2nd response."""
    rows = []
    for st, mat in ((0, 'Shading'), (1, 'Texture')):
        # In vs Out: R-C-L as separate units → 11 obj × 3 ori per material (33 each, 66 total)
        for obj in range(_n_obj):
            b_obj = perm[obj] if perm is not None else obj
            for ori in 'RCL':
                rows.append({
                    'Analysis': 'In_vs_Out', 'Material': mat,
                    'Object': obj + 1, 'Unit': ori,
                    'Unit_id': f'IO_{mat}_{obj + 1}_{ori}',
                    'Corr': _safe_corr(X[:, obj, st, _IN_K[ori]], X[:, b_obj, st, _OUT_K[ori]]),
                    'Condition': condition, 'Shuffle_id': shuffle_id,
                })
        # R-C-L: In/Out as separate units → 11 obj × 2 io per material (22 each)
        for obj in range(_n_obj):
            b_obj = perm[obj] if perm is not None else obj
            for io, ks in (('In', _IN_K), ('Out', _OUT_K)):
                cors = [_safe_corr(X[:, obj, st, ks[oa]], X[:, b_obj, st, ks[ob]])
                        for oa, ob in _ori_pairs]
                rows.append({
                    'Analysis': 'R_C_L', 'Material': mat,
                    'Object': obj + 1, 'Unit': io,
                    'Unit_id': f'CRL_{mat}_{obj + 1}_{io}',
                    'Corr': np.nanmean(cors),
                    'Condition': condition, 'Shuffle_id': shuffle_id,
                })
    # Shading vs Texture: 11 obj × 6 cond → 66 units
    for obj in range(_n_obj):
        b_obj = perm[obj] if perm is not None else obj
        for k, kl in enumerate(_K_LBL):
            rows.append({
                'Analysis': 'Shading_vs_Texture', 'Material': 'Across',
                'Object': obj + 1, 'Unit': kl,
                'Unit_id': f'ST_{obj + 1}_{k}',
                'Corr': _safe_corr(X[:, obj, 0, k], X[:, b_obj, 1, k]),
                'Condition': condition, 'Shuffle_id': shuffle_id,
            })
    return rows


_rows = _corr_rows(condition='real')
for si in range(_n_shuffle):
    _rows.extend(_corr_rows(rng.permutation(_n_obj), condition='shuffle', shuffle_id=si))

identity_corr_df = pd.DataFrame(_rows)
identity_corr_df

#%% Bar plot: paired t-test (real vs shuffle, paired by Unit_id)
from scipy import stats

_ana_keys = [
    ('In_vs_Out', 'Shading'), ('In_vs_Out', 'Texture'),
    ('R_C_L', 'Shading'), ('R_C_L', 'Texture'),
    ('Shading_vs_Texture', 'Across'),
]
_ana_lbl = {
    ('In_vs_Out', 'Shading'): 'In vs Out\n(Shading)',
    ('In_vs_Out', 'Texture'): 'In vs Out\n(Texture)',
    ('R_C_L', 'Shading'): 'R-C-L\n(Shading)',
    ('R_C_L', 'Texture'): 'R-C-L\n(Texture)',
    ('Shading_vs_Texture', 'Across'): 'Shading vs\nTexture',
}


def _paired_real_ctrl(df, ana, mat):
    real = df.loc[(df.Analysis == ana) & (df.Material == mat) & (df.Condition == 'real')].set_index('Unit_id')['Corr']
    ctrl = (df.loc[(df.Analysis == ana) & (df.Material == mat) & (df.Condition == 'shuffle')]
            .groupby('Unit_id')['Corr'].mean())
    idx = real.index.intersection(ctrl.index)
    return real.loc[idx], ctrl.loc[idx]


plot_rows, pvals = [], {}
for ana, mat in _ana_keys:
    real_s, ctrl_s = _paired_real_ctrl(identity_corr_df, ana, mat)
    _, pvals[(ana, mat)] = stats.ttest_rel(real_s, ctrl_s, nan_policy='omit')
    for uid in real_s.index:
        plot_rows.append({'Analysis': _ana_lbl[(ana, mat)], 'Unit_id': uid,
                          'Corr': real_s.loc[uid], 'Group': 'Real'})
        plot_rows.append({'Analysis': _ana_lbl[(ana, mat)], 'Unit_id': uid,
                          'Corr': ctrl_s.loc[uid], 'Group': 'Shuffle'})
corr_bar_df = pd.DataFrame(plot_rows)


def _p_to_star(p):
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


fig, ax = plt.subplots(figsize=(5, 4.2))
sns.barplot(
    data=corr_bar_df, x='Analysis', y='Corr', hue='Group',
    order=[_ana_lbl[k] for k in _ana_keys], hue_order=['Real', 'Shuffle'],
    estimator=np.mean, errorbar=('se', 1), capsize=0.08,
    palette={'Real': '#3b82f6', 'Shuffle': '#9ca3af'}, ax=ax,
)

real_bars, ctrl_bars = ax.containers[0], ax.containers[1]
for i, key in enumerate(_ana_keys):
    real_s, ctrl_s = _paired_real_ctrl(identity_corr_df, *key)
    x1 = real_bars[i].get_x() + real_bars[i].get_width() / 2
    x2 = ctrl_bars[i].get_x() + ctrl_bars[i].get_width() / 2
    for y1, y2 in zip(real_s, ctrl_s):
        ax.plot([x1, x2], [y1, y2], color='k', alpha=0.06, lw=0.6, zorder=1)

for i, key in enumerate(_ana_keys):
    b1, b2 = real_bars[i], ctrl_bars[i]
    x1 = b1.get_x() + b1.get_width() / 2
    x2 = b2.get_x() + b2.get_width() / 2
    y = max(b1.get_height(), b2.get_height()) + 0.03
    h = 0.015
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='k', lw=1)
    ax.text((x1 + x2) / 2, y + h + 0.005, _p_to_star(pvals[key]),
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('')
ax.set_ylabel('Pearson r')
ax.set_ylim(-0.15, 0.55)
ax.grid(axis='y', alpha=0.2, lw=0.8)
ax.legend(frameon=False, title='')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
plt.show()

#%%


