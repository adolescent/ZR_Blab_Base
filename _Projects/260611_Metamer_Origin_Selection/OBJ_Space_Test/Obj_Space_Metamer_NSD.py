'''
NSD 响应拟合偏好轴 + Shuffle 中介验证。

与 Test_Obj_Space_Rsp / Obj_Space_Shuffle_Intersected 的区别：
  - 样本空间仍用 NSD1k PCA（step1/step2 缓存，只读）
  - z-score 与 lstsq 拟合仅用 NSD 坐标 + NSD 响应
  - metamer 用于泛化 R² 与 shuffle 中介统计

依赖：
  Analysis/nsd1k_obj_space_step1.npz
  Analysis/metamer1k_obj_space_step2.npz
  Metamer_NSD_2k/{area}/avr_rsp.npy
  Metamer_NSD_2k/stim_layout.npz
'''

#%% 0. 配置
import os
import sys
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / 'Common_Functions'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import matplotlib.pyplot as plt
import OS_Tools as ot

metamer_nsd_path = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_NSD_2k'
analysis_root = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis'
savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\obj_space_metamer_nsd'
check_area = 'ASB'

N_DIM = 50
N_METAMER = 1000
N_NSD = 1000
N_OBJ = 40
N_SHUF = 5
SLICE_METAMER = slice(0, N_METAMER)
SLICE_NSD = slice(N_METAMER, N_METAMER + N_NSD)

COLOR_ANI = '#c0392b'
COLOR_INANI = '#2980b9'
SHUF_LABELS = ['Raw', 'S1', 'S2', 'S3', 'S4']
SHUF_ALPHA = [1.0, 0.62, 0.38, 0.20, 0.07]
SHUF_SIZE = [15] * 5
rsp_unit = 'spikes'

from obj_space_paths import area_dir, resolve_area_path, shared_path
from obj_space_plot import finish_fig, SCRIPT_METAMER_NSD

SAVE_FIGURES = True
SHOW_FIGURES = False


def _nsd_object_id_from_path(path):
    """Map NSD1000 filename to 1-based object id (50001.jpg -> 1)."""
    stem = Path(path).stem
    if not stem.isdigit():
        return None
    n = int(stem)
    if n >= 50000:
        return n - 50000
    return n


def _align_nsd_coords_and_rsp(nsd_coords, img_paths, rsp_nsd, nsd_object_ids):
    """
    Align step1 coords / Metamer_NSD responses by NSD object id (1..1000).
    Returns (coords_aligned, rsp_aligned, perm_info).
    """
    n_export = len(nsd_object_ids)
    assert rsp_nsd.shape[1] == n_export == N_NSD

    ids_from_paths = np.array([_nsd_object_id_from_path(p) for p in img_paths], dtype=np.int32)
    if np.any(ids_from_paths <= 0):
        raise ValueError('Could not parse NSD object ids from step1 img_paths')

    coords_by_oid = np.full((N_NSD, nsd_coords.shape[1]), np.nan, np.float32)
    for idx, oid in enumerate(ids_from_paths):
        if 1 <= oid <= N_NSD:
            coords_by_oid[oid - 1] = nsd_coords[idx]

    export_oids = nsd_object_ids.astype(np.int32)
    if not np.array_equal(export_oids, np.arange(1, N_NSD + 1)):
        perm = export_oids - 1
        rsp_aligned = rsp_nsd[:, perm]
        coords_aligned = coords_by_oid[export_oids - 1]
        reordered = True
    else:
        rsp_aligned = rsp_nsd
        coords_aligned = coords_by_oid
        reordered = False

    if np.any(~np.isfinite(coords_aligned)):
        missing = np.where(~np.isfinite(coords_aligned).any(1))[0] + 1
        raise ValueError(f'Missing NSD coords for object ids: {missing[:10]}...')

    return coords_aligned.astype(np.float32), rsp_aligned.astype(np.float32), reordered


def _r2(y, pred):
    v = y.var()
    if v <= 0:
        return np.nan
    return float(1.0 - (y - pred).var() / v)


#%% 1. 加载样本空间、响应，并对齐 NSD 顺序
CACHE_STEP1 = shared_path(analysis_root, 'step1')
CACHE_STEP2 = shared_path(analysis_root, 'step2')
RSP_PATH = ot.Join(ot.Join(metamer_nsd_path, check_area), 'avr_rsp.npy')
STIM_LAYOUT = ot.Join(metamer_nsd_path, 'stim_layout.npz')

d1 = np.load(CACHE_STEP1, allow_pickle=True)
d2 = np.load(CACHE_STEP2, allow_pickle=True)
layout = np.load(STIM_LAYOUT, allow_pickle=True)
rsp_all = np.load(RSP_PATH)

nsd_coords_raw = d1['coords'].astype(np.float32)
meta_coords = d2['coords'].astype(np.float32)
img_paths = list(d1['img_paths'])

rsp_meta = rsp_all[:, SLICE_METAMER].astype(np.float32)
rsp_nsd_raw = rsp_all[:, SLICE_NSD].astype(np.float32)

nsd_slice = layout['slice_nsd']
nsd_object_ids = layout['object_id'][nsd_slice[0]:nsd_slice[1]]
assert meta_coords.shape[0] == N_METAMER

nsd_coords, rsp_nsd, nsd_reordered = _align_nsd_coords_and_rsp(
    nsd_coords_raw, img_paths, rsp_nsd_raw, nsd_object_ids,
)

n_cell = rsp_all.shape[0]
print(f'{check_area}: {n_cell} cells')
print(f'NSD alignment reordered={nsd_reordered}  object_id range='
      f'[{nsd_object_ids.min()}, {nsd_object_ids.max()}]')

#%% 2. 用 NSD 响应拟合偏好轴
CACHE_FIT = ot.Join(area_dir(savepath, check_area), 'nsd_axis_fit.npz')

F_mu = nsd_coords.mean(0)
F_std = nsd_coords.std(0)
F_std[F_std < 1e-8] = 1.0
F_z_nsd = (nsd_coords - F_mu) / F_std
F_z_meta = (meta_coords - F_mu) / F_std
X_nsd = np.c_[F_z_nsd, np.ones(N_NSD)]

axes = np.zeros((n_cell, N_DIM), np.float32)
bias = np.zeros(n_cell, np.float32)
r2_nsd = np.zeros(n_cell, np.float32)
r2_meta = np.zeros(n_cell, np.float32)
nsd_load = np.zeros((n_cell, N_NSD), np.float32)
meta_load = np.zeros((n_cell, N_METAMER), np.float32)

for i in range(n_cell):
    coef, _, _, _ = np.linalg.lstsq(X_nsd, rsp_nsd[i], rcond=None)
    axes[i] = coef[:N_DIM]
    bias[i] = coef[N_DIM]
    nsd_load[i] = F_z_nsd @ axes[i]
    meta_load[i] = F_z_meta @ axes[i]
    r2_nsd[i] = _r2(rsp_nsd[i], nsd_load[i] + bias[i])
    r2_meta[i] = _r2(rsp_meta[i], meta_load[i] + bias[i])

ot.Mkdir(area_dir(savepath, check_area), mute=True)
np.savez(
    CACHE_FIT,
    axes=axes, bias=bias,
    r2_nsd=r2_nsd, r2_meta=r2_meta,
    nsd_load=nsd_load, meta_load=meta_load,
    F_mu=F_mu, F_std=F_std,
    check_area=np.array(check_area),
)

print(f'{check_area}: median R2_nsd={np.nanmedian(r2_nsd):.3f}  '
      f'median R2_meta={np.nanmedian(r2_meta):.3f}')
print(f'saved: {CACHE_FIT}')

legacy_fit = resolve_area_path(analysis_root, check_area, 'obj_axis_fit')
if os.path.isfile(legacy_fit):
    r2_legacy = np.load(legacy_fit, allow_pickle=True)['r2']
    print(f'[compare] legacy metamer-fit median R2={np.nanmedian(r2_legacy):.3f}  '
          f'vs NSD-fit R2_meta={np.nanmedian(r2_meta):.3f}')

#%% 3. 单神经元双面板拟合图 + 群体 R²
from matplotlib.lines import Line2D

demo_cell = 28

pc = int(demo_cell)
x_nsd = nsd_load[pc]
y_nsd = rsp_nsd[pc]
x_meta = meta_load[pc]
y_meta = rsp_meta[pc]

idx_meta = np.arange(N_METAMER)
within_meta = idx_meta % 200
shuffle_meta = within_meta // 40
is_ani_meta = (within_meta % 40) < 20

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))

# 左：NSD 训练集（灰色）
ax_l.scatter(x_nsd, y_nsd, s=8, alpha=0.28, c='0.75', edgecolors='none', zorder=1)
xl_nsd = np.linspace(x_nsd.min(), x_nsd.max(), 100)
m_nsd, b_nsd = np.polyfit(x_nsd, y_nsd, 1)
ax_l.plot(xl_nsd, m_nsd * xl_nsd + b_nsd, 'r-', lw=2, zorder=3)
ax_l.set_xlabel('Loading on preferred axis')
ax_l.set_ylabel(f'Response ({rsp_unit})')
ax_l.set_title(f'NSD (train)  R²={r2_nsd[pc]:.3f}')
ax_l.grid(True, ls=':', lw=0.5, alpha=0.4)

# 右：Metamer 泛化（ani/inani 配色 + shuffle 深浅）
for shuf in range(N_SHUF):
    for ani_flag, color in ((True, COLOR_ANI), (False, COLOR_INANI)):
        m = (shuffle_meta == shuf) & (is_ani_meta == ani_flag)
        ax_r.scatter(x_meta[m], y_meta[m], s=SHUF_SIZE[shuf], c=color,
                     alpha=SHUF_ALPHA[shuf], edgecolors='none', zorder=3 + shuf)

xl_meta = np.linspace(x_meta.min(), x_meta.max(), 100)
m_meta, b_meta = np.polyfit(x_meta, y_meta, 1)
ax_r.plot(xl_meta, m_meta * xl_meta + b_meta, 'k-', lw=1.5, zorder=2, alpha=0.7)
ax_r.set_xlabel('Loading on preferred axis')
ax_r.set_ylabel(f'Response ({rsp_unit})')
ax_r.set_title(f'Metamer (generalize)  R²={r2_meta[pc]:.3f}')
ax_r.grid(True, ls=':', lw=0.5, alpha=0.4)

leg_cat = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ANI, markersize=7, label='Ani'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INANI, markersize=7, label='Inani'),
]
leg_shuf = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
           markersize=SHUF_SIZE[s] ** 0.45, alpha=SHUF_ALPHA[s], label=SHUF_LABELS[s])
    for s in range(N_SHUF)
]
leg1 = ax_r.legend(handles=leg_cat, loc='upper left', fontsize=7, title='Category')
ax_r.add_artist(leg1)
ax_r.legend(handles=leg_shuf, loc='lower right', fontsize=7, title='Shuffle')

fig.suptitle(f'{check_area} cell {pc}  NSD-fitted preferred axis', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_METAMER_NSD, f'demo_cell{pc}_load_rsp_nsd_meta', area=check_area)

fig, axes_plt = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, vals, title in (
    (axes_plt[0], r2_nsd[np.isfinite(r2_nsd)], 'R²_nsd (train)'),
    (axes_plt[1], r2_meta[np.isfinite(r2_meta)], 'R²_meta (generalize)'),
):
    ax.hist(vals, bins=40, color='0.45', edgecolor='white', lw=0.5)
    ax.axvline(np.median(vals), color='C1', ls='--', lw=1.5,
               label=f'median={np.median(vals):.3f}')
    ax.set_xlabel('R²')
    ax.set_ylabel('Cell count')
    ax.set_title(title)
    ax.legend(fontsize=8)

fig.suptitle(f'{check_area}  NSD-fitted axis  N={n_cell}', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_METAMER_NSD, 'pop_r2_nsd_meta_hist', area=check_area)

fig, ax = plt.subplots(figsize=(4.5, 4))
m = np.isfinite(r2_nsd) & np.isfinite(r2_meta)
ax.scatter(r2_nsd[m], r2_meta[m], s=8, alpha=0.4, c='0.4', edgecolors='none', rasterized=True)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
ax.set_xlabel('R²_nsd (train)')
ax.set_ylabel('R²_meta (generalize)')
ax.set_title(f'{check_area}  train vs generalize')
ax.grid(True, ls=':', lw=0.5, alpha=0.4)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_METAMER_NSD, 'pop_r2_nsd_vs_meta', area=check_area)

#%% 4. Shuffle 中介 — 构建 avg_load / avg_rsp，计算斜率
meta_load = meta_load.astype(np.float32)
r2_load = r2_meta.astype(np.float32)
rsp = rsp_meta

idx = np.arange(N_METAMER)
within = idx % 200
shuffle = within // 40
is_ani = (within % 40) < 20
parent = within % 40

group_idx = np.full((N_OBJ, N_SHUF, N_METAMER // (N_OBJ * N_SHUF)), -1, dtype=int)
for o in range(N_OBJ):
    for s in range(N_SHUF):
        hits = np.where((parent == o) & (shuffle == s))[0]
        group_idx[o, s, :len(hits)] = hits

gi = group_idx.reshape(N_OBJ * N_SHUF, -1)
avg_load = meta_load[:, gi].mean(-1).reshape(n_cell, N_OBJ, N_SHUF)
avg_rsp = rsp[:, gi].mean(-1).reshape(n_cell, N_OBJ, N_SHUF)

shuf_c = np.arange(N_SHUF, dtype=np.float32) - 2.0
slope_load = (avg_load * shuf_c).sum(-1) / 10.0
slope_rsp = (avg_rsp * shuf_c).sum(-1) / 10.0

print(f'avg_load shape: {avg_load.shape}')
print(f'slope_load range: [{slope_load.min():.3f}, {slope_load.max():.3f}]')

#%% 5. Test 1 — 斜率散点 & 每神经元 Pearson r
from scipy import stats as sp_stats

sl_mu = slope_load.mean(1, keepdims=True)
sr_mu = slope_rsp.mean(1, keepdims=True)
sl_c = slope_load - sl_mu
sr_c = slope_rsp - sr_mu
cov = (sl_c * sr_c).sum(1)
std_l = np.sqrt((sl_c ** 2).sum(1))
std_r = np.sqrt((sr_c ** 2).sum(1))
denom = std_l * std_r
pearson_r = np.where(denom > 1e-8, cov / denom, np.nan)

print(f'Per-neuron Pearson r — median={np.nanmedian(pearson_r):.3f}  '
      f'mean={np.nanmean(pearson_r):.3f}  '
      f'frac>0.5: {np.nanmean(pearson_r > 0.5):.2%}')

fig, axes_fig = plt.subplots(1, 2, figsize=(11, 4.5))
is_ani_obj = np.arange(N_OBJ) < 20
sl_flat = slope_load.ravel()
sr_flat = slope_rsp.ravel()
ani_flat = np.tile(is_ani_obj, n_cell)

ax = axes_fig[0]
ax.scatter(sl_flat[ani_flat], sr_flat[ani_flat], s=4, alpha=0.15,
           c=COLOR_ANI, edgecolors='none', rasterized=True, label='Ani')
ax.scatter(sl_flat[~ani_flat], sr_flat[~ani_flat], s=4, alpha=0.15,
           c=COLOR_INANI, edgecolors='none', rasterized=True, label='Inani')

valid = np.isfinite(sl_flat) & np.isfinite(sr_flat)
lr = sp_stats.linregress(sl_flat[valid], sr_flat[valid])
xl = np.array([sl_flat[valid].min(), sl_flat[valid].max()])
ax.plot(xl, lr.slope * xl + lr.intercept, 'k-', lw=1.8, zorder=5, label='fit')
ax.plot(xl, xl, 'k--', lw=1.0, alpha=0.4, label='slope=1 (expected)')
ax.axhline(0, color='0.6', lw=0.6, ls=':')
ax.axvline(0, color='0.6', lw=0.6, ls=':')
ax.set_xlabel('slope_load  (Δload per shuffle unit)')
ax.set_ylabel(f'slope_rsp   (Δ{rsp_unit} per shuffle unit)')
ax.set_title(f'{check_area}  slope scatter  N={valid.sum()}')

stats_txt = (f'slope     = {lr.slope:.4f}\n'
             f'intercept = {lr.intercept:.5f}\n'
             f'R²        = {lr.rvalue**2:.4f}\n'
             f'p         = {lr.pvalue:.2e}')
ax.text(0.97, 0.03, stats_txt, transform=ax.transAxes,
        fontsize=8, va='bottom', ha='right', family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
ax.legend(fontsize=8, markerscale=3, loc='upper left')
ax.grid(True, ls=':', lw=0.5, alpha=0.4)

print(f'Global slope regression:  slope={lr.slope:.4f}  '
      f'R2={lr.rvalue**2:.4f}  p={lr.pvalue:.2e}')

ax = axes_fig[1]
valid_r = pearson_r[np.isfinite(pearson_r)]
ax.hist(valid_r, bins=40, color='0.45', edgecolor='white', lw=0.5)
ax.axvline(np.nanmedian(pearson_r), color='C1', ls='--', lw=1.5,
           label=f'median = {np.nanmedian(pearson_r):.3f}')
ax.axvline(0, color='k', ls=':', lw=1.0)
ax.set_xlabel('Pearson r  (slope_load vs slope_rsp,  40 objects)')
ax.set_ylabel('Cell count')
ax.set_title(f'{check_area}  per-neuron r  N={len(valid_r)}')
ax.legend(fontsize=8)

fig.suptitle('Test 1: Shuffle slope correlation (NSD-fitted axis)', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_METAMER_NSD, 'test1_slope_scatter', area=check_area)

#%% 6. Test 2 — 增量 ΔR²
shuf_f = shuffle.astype(np.float32)
shuf_z = (shuf_f - shuf_f.mean()) / (shuf_f.std() + 1e-8)

r2_full = np.full(n_cell, np.nan, np.float32)
for i in range(n_cell):
    X = np.c_[meta_load[i], shuf_z, np.ones(N_METAMER)]
    coef, _, _, _ = np.linalg.lstsq(X, rsp[i], rcond=None)
    pred = X @ coef
    ss_res = ((rsp[i] - pred) ** 2).sum()
    ss_tot = rsp[i].var() * N_METAMER
    r2_full[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

delta_r2 = r2_full - r2_load

print(f'dR2 = R2_full - R2_meta:')
print(f'  median = {np.nanmedian(delta_r2):.4f}')
print(f'  mean   = {np.nanmean(delta_r2):.4f}')
print(f'  frac < 0.01: {np.nanmean(delta_r2 < 0.01):.2%}')

fig, axes_fig = plt.subplots(1, 2, figsize=(9, 3.8))
ax = axes_fig[0]
dr = delta_r2[np.isfinite(delta_r2)]
ax.hist(dr, bins=40, color='0.45', edgecolor='white', lw=0.5)
ax.axvline(np.median(dr), color='C1', ls='--', lw=1.5,
           label=f'median={np.median(dr):.4f}')
ax.axvline(0, color='k', ls=':', lw=1.0)
ax.set_xlabel('ΔR²  (R²_full − R²_meta)')
ax.set_ylabel('Cell count')
ax.set_title('A: ΔR² distribution')
ax.legend(fontsize=8)

ax = axes_fig[1]
m = np.isfinite(delta_r2) & np.isfinite(r2_load)
ax.scatter(r2_load[m], delta_r2[m], s=8, alpha=0.4, c='0.4',
           edgecolors='none', rasterized=True)
ax.axhline(0, color='C1', ls='--', lw=1.0)
ax.set_xlabel('R²_meta  (NSD-fitted axis on metamer)')
ax.set_ylabel('ΔR²  (additional from shuffle)')
ax.set_title('B: R²_meta vs ΔR²')
ax.grid(True, ls=':', lw=0.5, alpha=0.4)

fig.suptitle(f'{check_area}  Test 2: Incremental ΔR² (NSD-fitted axis)', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_METAMER_NSD, 'test2_delta_r2', area=check_area)

CACHE_MEDIATION = ot.Join(area_dir(savepath, check_area), 'nsd_mediation.npz')
np.savez(
    CACHE_MEDIATION,
    slope_load=slope_load, slope_rsp=slope_rsp, pearson_r=pearson_r,
    avg_load=avg_load, avg_rsp=avg_rsp,
    r2_full=r2_full, delta_r2=delta_r2,
    r2_meta=r2_load,
)
print(f'saved: {CACHE_MEDIATION}')

#%% 7a. Demo 筛选 — ani / inani 分别排名
demo_ani_obj = 3
demo_inani_obj = 26
slope_sign = +1
top_n_print = 10

al_c = avg_load - avg_load.mean(-1, keepdims=True)
ar_c = avg_rsp - avg_rsp.mean(-1, keepdims=True)
cor_per_obj = np.clip(
    (al_c * ar_c).sum(-1) / (
        np.sqrt((al_c ** 2).sum(-1)) * np.sqrt((ar_c ** 2).sum(-1)) + 1e-16
    ), -1, 1,
)

dir_label6 = ('反例 (↑load↑rsp)' if slope_sign > 0 else
              '正例 (↓load↓rsp)' if slope_sign < 0 else '不限方向')


def _rank_for_obj(obj_idx, sign):
    sc = cor_per_obj[:, obj_idx] * r2_load
    sc = sc.copy().astype(float)
    sc[~np.isfinite(sc)] = np.nan
    if sign != 0:
        sc[slope_load[:, obj_idx] * sign <= 0] = np.nan
    return np.argsort(np.nan_to_num(sc, nan=-np.inf))[::-1], sc


ranked_ani, score_ani = _rank_for_obj(demo_ani_obj, slope_sign)
ranked_inani, score_inani = _rank_for_obj(demo_inani_obj, slope_sign)

for tag, ranked, score, obj_idx in [
    ('ANI', ranked_ani, score_ani, demo_ani_obj),
    ('INANI', ranked_inani, score_inani, demo_inani_obj),
]:
    print(f'\n── {tag} obj {obj_idx + 1}  [{dir_label6}] ──')
    print(f'{"rank":>4}  {"cell":>6}  {"score":>7}  {"R2_meta":>8}  '
          f'{"cor":>7}  {"slope_load":>11}  {"slope_rsp":>10}')
    for rank, ci in enumerate(ranked[:top_n_print], 1):
        if np.isnan(score[ci]):
            break
        print(f'{rank:>4}  {ci:>6}  {score[ci]:>7.4f}  {r2_load[ci]:>8.3f}  '
              f'{cor_per_obj[ci, obj_idx]:>7.3f}  '
              f'{slope_load[ci, obj_idx]:>11.4f}  '
              f'{slope_rsp[ci, obj_idx]:>10.4f}')

cor_ani_flat = cor_per_obj[:, :20].ravel()
cor_inani_flat = cor_per_obj[:, 20:].ravel()
cor_ani_flat = cor_ani_flat[np.isfinite(cor_ani_flat)]
cor_inani_flat = cor_inani_flat[np.isfinite(cor_inani_flat)]

fig, ax = plt.subplots(figsize=(5.5, 3.8))
bins = np.linspace(-1, 1, 35)
ax.hist(cor_ani_flat, bins=bins, color=COLOR_ANI, alpha=0.55, edgecolor='white',
        lw=0.5, label=f'Ani  (median={np.median(cor_ani_flat):.3f})')
ax.hist(cor_inani_flat, bins=bins, color=COLOR_INANI, alpha=0.55, edgecolor='white',
        lw=0.5, label=f'Inani (median={np.median(cor_inani_flat):.3f})')
ax.axvline(np.median(cor_ani_flat), color=COLOR_ANI, ls='--', lw=1.5)
ax.axvline(np.median(cor_inani_flat), color=COLOR_INANI, ls='--', lw=1.5)
ax.axvline(0, color='k', ls=':', lw=0.8)
ax.set_xlabel('Pearson r  (load ~ rsp,  5 shuffle points,  per neuron × object)')
ax.set_ylabel('Count  (neuron × object pairs)')
ax.set_title(f'{check_area}  Load–Rsp coupling (NSD-fitted axis)')
ax.legend(fontsize=9)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_METAMER_NSD, 'test6a_cor_coupling', area=check_area)

# 7b. Demo 单细胞图 — ani / inani 最佳神经元
rank_ani = 2
rank_inani = 1
cell_ani = None
cell_inani = None

pc_ani = int(cell_ani) if cell_ani is not None else int(ranked_ani[rank_ani - 1])
pc_inani = int(cell_inani) if cell_inani is not None else int(ranked_inani[rank_inani - 1])
shuf_x = np.arange(N_SHUF)


def _single_cell_plot(pc, obj_idx, color, tag):
    img_id = obj_idx + 1
    print(f'{tag} — cell {pc}  img {img_id}  R2_meta={r2_load[pc]:.3f}  '
          f'cor={cor_per_obj[pc, obj_idx]:.3f}  '
          f'sl_load={slope_load[pc, obj_idx]:.4f}  '
          f'sl_rsp={slope_rsp[pc, obj_idx]:.4f}')
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, data, ylabel in zip(
        [ax_l, ax_r],
        [avg_load, avg_rsp],
        ['Load on preferred axis', f'Firing rate ({rsp_unit})'],
    ):
        ax.plot(shuf_x, data[pc, obj_idx, :], '-o', color=color,
                lw=2.2, ms=8, label=f'{tag} img {img_id}')
        ax.set_xticks(shuf_x)
        ax.set_xticklabels(SHUF_LABELS)
        ax.set_xlabel('Shuffle level')
        ax.set_ylabel(ylabel)
        ax.grid(True, ls=':', lw=0.5, alpha=0.4)
        ax.legend(fontsize=9)
    ax_l.set_title('Load on preferred axis')
    ax_r.set_title('Firing rate')
    fig.suptitle(
        f'{check_area} cell {pc}  img {img_id}  R²_meta={r2_load[pc]:.3f}  (NSD-fitted axis)',
        fontsize=10,
    )
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_METAMER_NSD, f'demo_{tag.lower()}_cell{pc}_obj{img_id}', area=check_area)


_single_cell_plot(pc_ani, demo_ani_obj, COLOR_ANI, 'Ani')
_single_cell_plot(pc_inani, demo_inani_obj, COLOR_INANI, 'Inani')

#%%
