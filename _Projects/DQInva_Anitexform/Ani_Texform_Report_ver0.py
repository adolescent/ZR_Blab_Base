'''

Similarity analysis between real obj and texform obj.

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
texform_stims = np.load(ot.Join(save_path,'Ani_Texform_Redplot_z.npz'),allow_pickle=True)['redplot_z']

#%% RSA: 480×480, reordered AniB–AniS–ObjB–ObjS per block
# orig per 120: AniB | ObjB | AniS | ObjS  →  plot: AniB | AniS | ObjB | ObjS
_blk4 = lambda s: np.r_[s + np.arange(30), s + 60 + np.arange(30),
                        s + 30 + np.arange(30), s + 90 + np.arange(30)]
idx = np.concatenate([_blk4(s) for s in (0, 120, 240, 360)])
rsa = np.corrcoef(texform_stims[:, idx].T)
np.fill_diagonal(rsa, 0)

_sub_lbl = ['AB', 'AS', 'OB', 'OS']
_blk_lbl = ['Low Real', 'Low Texform', 'High Real', 'High Texform']

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(rsa, vmin=-0.4, vmax=0.4, center=0, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, square=True, ax=ax, cbar=False)
ax.axvline(240, color='k', lw=2)
ax.axhline(240, color='k', lw=2)
for x in (120, 360):
    ax.axvline(x, color='C0', lw=2)
    ax.axhline(x, color='C0', lw=2)
for b in (0, 120, 240, 360):
    for x in (b + 30, b + 60, b + 90):
        ax.axvline(x, color='yellow', lw=1)
        ax.axhline(x, color='yellow', lw=1)
for bi, b in enumerate((0, 120, 240, 360)):
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
ax.set_xlabel('Low resolution  |  High resolution', fontsize=10)
# ax.set_ylabel('Low resolution  |  High resolution', fontsize=10)
fig.tight_layout()

#%% per-object paired r (120 objs × 6 contrasts × matched / shuffle control)
R = texform_stims
_groups = {
    'ani_big': np.arange(0, 30), 'ani_small': np.arange(60, 90),
    'obj_big': np.arange(30, 60), 'obj_small': np.arange(90, 120),
}
_o2grp = {o: g for g, idx in _groups.items() for o in idx}
_pairs = [
    ('H_L', lambda o: (240 + o, o)),
    ('H_HTx', lambda o: (240 + o, 360 + o)),
    ('L_LTx', lambda o: (o, 120 + o)),
    ('HTx_LTx', lambda o: (360 + o, 120 + o)),
    ('HTx_L', lambda o: (360 + o, o)),
    ('LTx_H', lambda o: (120 + o, 240 + o)),
]
_rng = np.random.default_rng(0)
_rows = []
for o in range(120):
    for pname, fn in _pairs:
        i, j = fn(o)
        r_m = np.corrcoef(R[:, i], R[:, j])[0, 1]
        r_sh = [np.corrcoef(R[:, fn(o)[0]], R[:, fn(_rng.permutation(120)[o])[1]])[0, 1]
                for _ in range(10)]
        _rows.append(dict(Object=o + 1, Group=_o2grp[o], Corr_type=pname,
                          Match='matched', Corr=r_m))
        _rows.append(dict(Object=o + 1, Group=_o2grp[o], Corr_type=pname,
                          Match='shuffle', Corr=np.mean(r_sh)))
obj_cond_corr = pd.DataFrame(_rows)

#%% bar plot (order: H–L … LTx–H); sig: HTx–LTx vs LTx–H & HTx–L
from scipy import stats
from matplotlib.patches import Patch

_corr_order = ['H_L', 'H_HTx', 'L_LTx', 'HTx_LTx', 'HTx_L', 'LTx_H']
_corr_short = ['H–L', 'H–HTx', 'L–LTx', 'HTx–LTx', 'HTx–L', 'LTx–H']
_sig_pairs = [('HTx_LTx', 'LTx_H'), ('HTx_LTx', 'HTx_L')]
_grp_order = ['ani_big', 'ani_small', 'obj_big', 'obj_small']
_palette = sns.color_palette('colorblind', 6)
_n = len(_corr_order)
_x0, _w = np.arange(4), 0.11
_off = (np.arange(_n) - (_n - 1) / 2) * _w
_bw, _dx = _w * 0.36, _w * 0.11

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='#ccc', lw=0.8, zorder=0)
for i, grp in enumerate(_grp_order):
    d = obj_cond_corr.query('Group == @grp')
    mat = d.query('Match == "matched"').pivot(index='Object', columns='Corr_type', values='Corr')[_corr_order]
    shu = d.query('Match == "shuffle"').pivot(index='Object', columns='Corr_type', values='Corr')[_corr_order]
    x0 = _x0[i] + _off
    ax.bar(x0 - _dx, shu.mean(), _bw, yerr=shu.sem(), color=_palette, alpha=0.3,
           capsize=2, error_kw=dict(lw=0.8), linewidth=0)
    ax.bar(x0 + _dx, mat.mean(), _bw, yerr=mat.sem(), color=_palette, capsize=2,
           error_kw=dict(lw=0.8), zorder=3)
    y0 = max((mat.mean() + mat.sem()).max(), (shu.mean() + shu.sem()).max()) + 0.025
    for ki, (a, b) in enumerate(_sig_pairs):
        p = stats.ttest_rel(mat[a], mat[b]).pvalue
        if p >= 0.05:
            continue
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
        ia, ib = _corr_order.index(a), _corr_order.index(b)
        xa, xb = x0[ia] + _dx, x0[ib] + _dx
        yb = y0 + ki * 0.055
        ax.plot([xa, xa, xb, xb], [yb, yb + 0.01, yb + 0.01, yb], 'k-', lw=0.85, zorder=4)
        ax.text((xa + xb) / 2, yb + 0.01, star, ha='center', va='bottom', fontsize=10)

ax.set_xticks(_x0, _grp_order, fontsize=11)
ax.set_ylabel('Pearson r', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(_palette, _corr_short)]
          + [Patch(facecolor='gray', alpha=0.3, label='Shuffle'),
             Patch(facecolor='gray', label='Matched')],
          frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.8, 1.14))
fig.tight_layout()

#%%



