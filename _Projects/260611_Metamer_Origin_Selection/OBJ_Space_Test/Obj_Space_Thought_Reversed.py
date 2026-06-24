'''
逆转思路：在已有 50D 样本空间中，用 linear fit 找反映 metamer 打乱程度的轴向。
分别对 ani / inani：shuffle level ~ 相对 Raw 的位移在轴上的投影。
'''

#%% 配置 & 加载样本空间（复用 step1/2 缓存，不重建）
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import OS_Tools as ot
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import matplotlib.colors as mcolors

savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis'
metamer_figpath = r'E:\#Stimsets\Metamer_P4_C4321_Object_STI150_1300'
check_area = 'ASB'
cell_rootpath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
N_DIM = 50
N_METAMER = 1000

from obj_space_paths import area_path, resolve_area_path, rsp_path, shared_path
from obj_space_plot import finish_fig, SCRIPT_THOUGHT

SAVE_FIGURES = True
SHOW_FIGURES = True

CACHE = shared_path(savepath, 'step1')
CACHE2 = shared_path(savepath, 'step2')

d1 = np.load(CACHE, allow_pickle=True)
d2 = np.load(CACHE2, allow_pickle=True)
nsd_coords = d1['coords']
meta_coords = d2['coords']
nsd_img_paths = list(d1['img_paths'])
meta_paths = list(d2['img_paths'])

# --- labels ---
idx = np.arange(N_METAMER)
within = idx % 200
shuffle = within // 40          # 0=Raw, 1–4=S1–S4
is_ani = (within % 40) < 20
parent_id = within % 40         # 0–39

COLOR_ANI = '#c0392b'
COLOR_INANI = '#2980b9'
SHUF_LABELS = ['Raw', 'S1', 'S2', 'S3', 'S4']
SHUF_ALPHA = [1.0, 0.72, 0.52, 0.34, 0.18]   # plot2 仍用透明度编码 shuffle
SHUF_SAT = [0.22, 0.42, 0.62, 0.82, 1.0]      # plot3：饱和度编码 shuffle level


def _rainbow_parent_colors(n_obj):
    """parent identity → rainbow 上均匀采样。"""
    return [plt.cm.rainbow(t)[:3] for t in np.linspace(0, 1, n_obj)]


def _color_by_shuffle(parent_rgb, shuffle_level, sat_levels=SHUF_SAT):
    """固定 parent 色相，用饱和度表示 shuffle level。"""
    h, _, v = mcolors.rgb_to_hsv(parent_rgb)
    return mcolors.hsv_to_rgb([h, sat_levels[shuffle_level], v])

#%% 拟合 shuffle 轴（ani / inani 各一根，50D 原空间）
def _fit_shuffle_axis(meta_coords, obj_ids, n_dim=N_DIM):
    """
    对每个 parent object，取 Raw→S4 共 5 点；feat = coord - coord_raw，y = shuffle level。
    在组内所有 object 上 pooled lstsq：y = feat @ w（过原点）。
    """
    feats, ys = [], []
    per_obj_raw = {}
    for obj in obj_ids:
        raw_idx = obj
        raw = meta_coords[raw_idx, :n_dim]
        per_obj_raw[obj] = raw
        for s in range(5):
            i = obj + s * 40
            feats.append(meta_coords[i, :n_dim] - raw)
            ys.append(float(s))
    F = np.asarray(feats, np.float64)
    y = np.asarray(ys, np.float64)
    w, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
    pred = F @ w
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_group = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return w.astype(np.float32), r2_group, per_obj_raw


def _loadings(meta_coords, w, per_obj_raw, obj_ids, n_dim=N_DIM):
    """沿轴载荷：Raw 点 = 0，其余为 (coord - raw) @ w。"""
    loads = np.zeros(N_METAMER, np.float32)
    for obj in obj_ids:
        for s in range(5):
            i = obj + s * 40
            loads[i] = (meta_coords[i, :n_dim] - per_obj_raw[obj]) @ w
    return loads


def _per_obj_r2(loads, obj_ids):
    """每个 object 上 shuffle ~ loading 的 R²（沿共享轴）。"""
    r2s = []
    for obj in obj_ids:
        ls = np.array([loads[obj + s * 40] for s in range(5)])
        sh = np.arange(5, dtype=float)
        if ls.var() < 1e-12:
            r2s.append(np.nan)
            continue
        c = np.corrcoef(sh, ls)[0, 1]
        r2s.append(float(c ** 2))
    return np.array(r2s)


ani_objs = list(range(20))
inani_objs = list(range(20, 40))
all_objs = list(range(40))
COLOR_ALL = '#2c3e50'

w_ani, r2_ani, raw_ani = _fit_shuffle_axis(meta_coords, ani_objs)
w_inani, r2_inani, raw_inani = _fit_shuffle_axis(meta_coords, inani_objs)
w_all, r2_all, raw_all = _fit_shuffle_axis(meta_coords, all_objs)

load_ani = _loadings(meta_coords, w_ani, raw_ani, ani_objs)
load_inani = _loadings(meta_coords, w_inani, raw_inani, inani_objs)
load_all = _loadings(meta_coords, w_all, raw_all, all_objs)

r2_ani_each = _per_obj_r2(load_ani, ani_objs)
r2_inani_each = _per_obj_r2(load_inani, inani_objs)
r2_all_each = _per_obj_r2(load_all, all_objs)

print(f'Ani    group R² = {r2_ani:.3f}   per-object median R² = {np.nanmedian(r2_ani_each):.3f}')
print(f'Inani  group R² = {r2_inani:.3f}   per-object median R² = {np.nanmedian(r2_inani_each):.3f}')
print(f'All    group R² = {r2_all:.3f}   per-object median R² = {np.nanmedian(r2_all_each):.3f}')

#%% plot1. R² 分布（组内 pooled + 逐 object）
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

for ax, r2_each, r2_g, label, color in zip(
    axes,
    [r2_ani_each, r2_inani_each],
    [r2_ani, r2_inani],
    ['Ani', 'Inani'],
    [COLOR_ANI, COLOR_INANI],
):
    valid = r2_each[np.isfinite(r2_each)]
    ax.hist(valid, bins=12, color=color, alpha=0.55, edgecolor='white', lw=0.6)
    ax.axvline(np.median(valid), color='k', ls='--', lw=1.2,
               label=f'obj median = {np.median(valid):.3f}')
    ax.axvline(r2_g, color='C1', ls=':', lw=1.5, label=f'group R² = {r2_g:.3f}')
    ax.set_xlabel('R² (shuffle ~ axis loading)')
    ax.set_ylabel('Object count')
    ax.set_title(f'{label}  (N_obj = {len(r2_each)})')
    ax.legend(fontsize=8)
fig.suptitle('Shuffle axis linear fit (50D object space)', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'shuffle_r2_ani_inani')

fig, ax = plt.subplots(figsize=(4, 3.2))
valid = r2_all_each[np.isfinite(r2_all_each)]
ax.hist(valid, bins=16, color=COLOR_ALL, alpha=0.55, edgecolor='white', lw=0.6)
ax.axvline(np.median(valid), color='k', ls='--', lw=1.2,
           label=f'obj median = {np.median(valid):.3f}')
ax.axvline(r2_all, color='C1', ls=':', lw=1.5, label=f'group R² = {r2_all:.3f}')
ax.set_xlabel('R² (shuffle ~ axis loading)')
ax.set_ylabel('Object count')
ax.set_title(f'All  (N_obj = {len(all_objs)})')
ax.legend(fontsize=8)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'shuffle_r2_all')

#%% plot2. PC1–PC2 上的刺激分布 + 轴方向（第一 cycle，200 点）
plot_idx = idx < 200

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, w, loads, obj_ids, color_base, title in zip(
    axes,
    [w_ani, w_inani],
    [load_ani, load_inani],
    [ani_objs, inani_objs],
    [COLOR_ANI, COLOR_INANI],
    ['Ani shuffle axis', 'Inani shuffle axis'],
):
    ax.scatter(nsd_coords[:, 0], nsd_coords[:, 1], s=5, c='#dddddd',
               edgecolors='none', rasterized=True, zorder=1)
    for shuf, lab, alpha in zip(range(5), SHUF_LABELS, SHUF_ALPHA):
        m = plot_idx & (shuffle == shuf) & np.isin(parent_id, obj_ids)
        ax.scatter(meta_coords[m, 0], meta_coords[m, 1], s=22, c=color_base, alpha=alpha,
                   edgecolors='white', linewidths=0.25, zorder=3 + shuf, rasterized=True)

    # 轴在 PC1–PC2 平面的投影（过各 object Raw 点的平均位置）
    w2 = w[:2]
    if np.linalg.norm(w2) > 1e-8:
        u2 = w2 / np.linalg.norm(w2)
        raw_mean = np.mean([meta_coords[o, :2] for o in obj_ids], axis=0)
        span = np.ptp(meta_coords[plot_idx & np.isin(parent_id, obj_ids), :2]) * 0.35
        ax.annotate('', xy=raw_mean + u2 * span, xytext=raw_mean,
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.grid(True, ls=':', lw=0.5, alpha=0.4)

leg_shuf = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', markersize=7,
           alpha=a, label=lab)
    for lab, a in zip(SHUF_LABELS, SHUF_ALPHA)
]
axes[0].legend(handles=leg_shuf, loc='upper left', fontsize=8, title='Shuffle')
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'shuffle_axis_pc12_ani_inani')

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(nsd_coords[:, 0], nsd_coords[:, 1], s=5, c='#dddddd',
           edgecolors='none', rasterized=True, zorder=1)
for shuf, alpha in zip(range(5), SHUF_ALPHA):
    for ani_flag, color in ((True, COLOR_ANI), (False, COLOR_INANI)):
        m = plot_idx & (shuffle == shuf) & (is_ani == ani_flag)
        ax.scatter(meta_coords[m, 0], meta_coords[m, 1], s=22, c=color, alpha=alpha,
                   edgecolors='white', linewidths=0.25, zorder=3 + shuf, rasterized=True)
w2 = w_all[:2]
if np.linalg.norm(w2) > 1e-8:
    u2 = w2 / np.linalg.norm(w2)
    raw_mean = np.mean(meta_coords[all_objs, :2], axis=0)
    span = np.ptp(meta_coords[plot_idx, :2]) * 0.35
    ax.annotate('', xy=raw_mean + u2 * span, xytext=raw_mean,
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('All shuffle axis (40 obj)')
ax.grid(True, ls=':', lw=0.5, alpha=0.4)
leg_cat = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ANI, markersize=7, label='Ani'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INANI, markersize=7, label='Inani'),
]
ax.legend(handles=leg_cat + leg_shuf, loc='upper left', fontsize=8)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'shuffle_axis_pc12_all')

#%% plot3. 沿 shuffle 轴的载荷分布（类似单细胞 tuning 图，第一 cycle 200 点）
n_extreme = 5

def _plot_axis_tuning(w, loads, obj_ids, r2_g, title_prefix):
    m_cycle = plot_idx & np.isin(parent_id, obj_ids)
    x = loads[m_cycle]
    y = shuffle[m_cycle].astype(float)
    n_obj = len(obj_ids)
    obj_rank = {obj: i for i, obj in enumerate(obj_ids)}
    parent_colors = _rainbow_parent_colors(n_obj)

    fig = plt.figure(figsize=(7.5, 5.5))
    gs = fig.add_gridspec(n_extreme, 3, width_ratios=[1, 3.2, 1], hspace=0.12, wspace=0.08)
    ax = fig.add_subplot(gs[:, 1])
    axes_lo = [fig.add_subplot(gs[i, 0]) for i in range(n_extreme)]
    axes_hi = [fig.add_subplot(gs[i, 2]) for i in range(n_extreme)]

    for obj in obj_ids:
        rank = obj_rank[obj]
        base = parent_colors[rank]
        for s in range(5):
            stim = obj + s * 40
            if not plot_idx[stim]:
                continue
            c = _color_by_shuffle(base, s)
            ax.scatter(loads[stim], shuffle[stim], s=28, c=[c],
                       edgecolors='white', linewidths=0.35, zorder=3 + s)

    xl = np.linspace(x.min(), x.max(), 100)
    coef = np.polyfit(x, y, 1)
    ax.plot(xl, np.polyval(coef, xl), 'k-', lw=1.5, alpha=0.75, zorder=2)

    # 极端 parent：按 S4 载荷排序（Raw 载荷恒为 0）
    load_s4 = np.array([loads[o + 4 * 40] for o in obj_ids])
    order = np.argsort(load_s4)
    lo_parents = [obj_ids[i] for i in order[:n_extreme]]
    hi_parents = [obj_ids[i] for i in order[-n_extreme:][::-1]]

    for i, p in enumerate(lo_parents):
        stim = int(p)  # Raw
        edge_c = _color_by_shuffle(parent_colors[obj_rank[p]], 0)
        axes_lo[i].imshow(Image.open(meta_paths[stim]))
        axes_lo[i].set_xticks([])
        axes_lo[i].set_yticks([])
        for sp in axes_lo[i].spines.values():
            sp.set_edgecolor(edge_c)
            sp.set_linewidth(1.2)
        fig.add_artist(ConnectionPatch(
            (1.0, 0.5), (loads[stim], 0), 'axes fraction', 'data',
            axesA=axes_lo[i], axesB=ax, color=edge_c, lw=0.7, alpha=0.55))
    for i, p in enumerate(hi_parents):
        stim = int(p + 4 * 40)  # S4
        edge_c = _color_by_shuffle(parent_colors[obj_rank[p]], 4)
        axes_hi[i].imshow(Image.open(meta_paths[stim]))
        axes_hi[i].set_xticks([])
        axes_hi[i].set_yticks([])
        for sp in axes_hi[i].spines.values():
            sp.set_edgecolor(edge_c)
            sp.set_linewidth(1.2)
        fig.add_artist(ConnectionPatch(
            (0.0, 0.5), (loads[stim], 4), 'axes fraction', 'data',
            axesA=axes_hi[i], axesB=ax, color=edge_c, lw=0.7, alpha=0.55))

    ax.set_xlabel('Loading on shuffle axis')
    ax.set_ylabel('Shuffle level')
    ax.set_yticks(range(5))
    ax.set_yticklabels(SHUF_LABELS)
    ax.set_title(f'{title_prefix}  group R²={r2_g:.3f}')
    ref_rgb = plt.cm.rainbow(0.5)[:3]
    leg_shuf = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=_color_by_shuffle(ref_rgb, s),
               markersize=7, label=SHUF_LABELS[s])
        for s in range(5)
    ]
    ax.legend(handles=leg_shuf, loc='lower right', fontsize=7, title='Shuffle',
              framealpha=0.92)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, f'shuffle_tuning_{title_prefix.lower().split()[0]}')


_plot_axis_tuning(w_ani, load_ani, ani_objs, r2_ani, 'Ani')
_plot_axis_tuning(w_inani, load_inani, inani_objs, r2_inani, 'Inani')
_plot_axis_tuning(w_all, load_all, all_objs, r2_all, 'All')

#%% ani / inani shuffle 轴夹角
# w_ani、w_inani：50D 样本空间中，各自类别 pooled linear fit 得到的 shuffle 敏感方向
u_ani = w_ani / (np.linalg.norm(w_ani) + 1e-8)
u_inani = w_inani / (np.linalg.norm(w_inani) + 1e-8)
cos_ang = float(np.clip(u_ani @ u_inani, -1.0, 1.0))
angle_deg = float(np.degrees(np.arccos(cos_ang)))
print(f'w_ani ∥ w_inani: angle = {angle_deg:.1f}°   cos = {cos_ang:.4f}')
u_all = w_all / (np.linalg.norm(w_all) + 1e-8)
print(f'w_all ∥ w_ani:   angle = {np.degrees(np.arccos(np.clip(u_all @ u_ani, -1, 1))):.1f}°')
print(f'w_all ∥ w_inani: angle = {np.degrees(np.arccos(np.clip(u_all @ u_inani, -1, 1))):.1f}°')

#%% 4.神经元 × shuffle 轴：夹角 & response~载荷 R²（重跑此 cell 才重算）
import pandas as pd

check_area = 'ASB'
cell_rootpath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
CACHE3 = resolve_area_path(savepath, check_area, 'obj_axis_fit')
CACHE_NEURON = area_path(savepath, check_area, 'shuffle_neuron')
SUMMARY_NEURON = area_path(savepath, check_area, 'shuffle_neuron_summary')

if not os.path.isfile(CACHE3):
    raise FileNotFoundError(
        f'{CACHE3} not found; run Test_Obj_Space_Rsp.py §3 or All_Brain_Areas.py first'
    )

d3 = np.load(CACHE3, allow_pickle=True)
axes_cell = d3['axes']
F_mu, F_std = d3['F_mu'], d3['F_std']
r2_obj_axis = d3['r2']
rsp = np.load(rsp_path(cell_rootpath, check_area))
n_cell = rsp.shape[0]
assert axes_cell.shape[0] == n_cell


def _batch_lin_r2(x, Y):
    """x (n_stim,), Y (n_cell, n_stim) → R² per cell."""
    x = np.asarray(x, np.float64)
    Y = np.asarray(Y, np.float64)
    X = np.c_[x, np.ones(len(x))]
    coef, _, _, _ = np.linalg.lstsq(X, Y.T, rcond=None)
    pred = X @ coef
    ss_res = ((Y.T - pred) ** 2).sum(0)
    ss_tot = ((Y.T - Y.T.mean(0)) ** 2).sum(0)
    return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan).astype(np.float32)


mask_ani = is_ani
mask_inani = ~is_ani
mask_all = np.ones(N_METAMER, dtype=bool)

# shuffle 轴转到 z-space，与神经元偏好轴一致
w_ani_z = w_ani / F_std
w_inani_z = w_inani / F_std
w_all_z = w_all / F_std
u_shuf_ani = w_ani_z / (np.linalg.norm(w_ani_z) + 1e-8)
u_shuf_inani = w_inani_z / (np.linalg.norm(w_inani_z) + 1e-8)
u_shuf_all = w_all_z / (np.linalg.norm(w_all_z) + 1e-8)
u_cell = axes_cell / (np.linalg.norm(axes_cell, axis=1, keepdims=True) + 1e-8)

cos_ani = np.clip(u_cell @ u_shuf_ani, -1.0, 1.0).astype(np.float32)
cos_inani = np.clip(u_cell @ u_shuf_inani, -1.0, 1.0).astype(np.float32)
cos_all = np.clip(u_cell @ u_shuf_all, -1.0, 1.0).astype(np.float32)
angle_ani = np.degrees(np.arccos(cos_ani)).astype(np.float32)
angle_inani = np.degrees(np.arccos(cos_inani)).astype(np.float32)
angle_all = np.degrees(np.arccos(cos_all)).astype(np.float32)

r2_shuf_ani = _batch_lin_r2(load_ani[mask_ani], rsp[:, mask_ani])
r2_shuf_inani = _batch_lin_r2(load_inani[mask_inani], rsp[:, mask_inani])
r2_shuf_all = _batch_lin_r2(load_all, rsp)

summary = pd.DataFrame({
    'cell_idx': np.arange(n_cell),
    'angle_ani': angle_ani,
    'angle_inani': angle_inani,
    'angle_all': angle_all,
    'cos_ani': cos_ani,
    'cos_inani': cos_inani,
    'cos_all': cos_all,
    'r2_shuf_ani': r2_shuf_ani,
    'r2_shuf_inani': r2_shuf_inani,
    'r2_shuf_all': r2_shuf_all,
    'r2_obj_axis': r2_obj_axis,
})
from obj_space_paths import area_dir
ot.Mkdir(area_dir(savepath, check_area), mute=True)
np.savez(CACHE_NEURON, angle_ani=angle_ani, angle_inani=angle_inani, angle_all=angle_all,
         cos_ani=cos_ani, cos_inani=cos_inani, cos_all=cos_all,
         r2_shuf_ani=r2_shuf_ani, r2_shuf_inani=r2_shuf_inani, r2_shuf_all=r2_shuf_all,
         check_area=np.array(check_area))
summary.to_csv(SUMMARY_NEURON, index=False)

print(f'{check_area}: {n_cell} cells')
print(f'  r2_shuf_ani   median={np.nanmedian(r2_shuf_ani):.3f}  mean={np.nanmean(r2_shuf_ani):.3f}')
print(f'  r2_shuf_inani median={np.nanmedian(r2_shuf_inani):.3f}  mean={np.nanmean(r2_shuf_inani):.3f}')
print(f'  r2_shuf_all   median={np.nanmedian(r2_shuf_all):.3f}  mean={np.nanmean(r2_shuf_all):.3f}')
print(f'  angle_ani     median={np.nanmedian(angle_ani):.1f}°')
print(f'  angle_inani   median={np.nanmedian(angle_inani):.1f}°')
print(f'  angle_all     median={np.nanmedian(angle_all):.1f}°')
print(f'saved: {CACHE_NEURON}')
print(f'saved: {SUMMARY_NEURON}')

#%% 4plot.群体统计：R² / 夹角（只读 shuffle_neuron）
dn = np.load(resolve_area_path(savepath, check_area, 'shuffle_neuron'), allow_pickle=True)
r2_a = dn['r2_shuf_ani']
r2_i = dn['r2_shuf_inani']
ang_a = dn['angle_ani']
ang_i = dn['angle_inani']
area = str(dn['check_area'])

# --- fig A: 每神经元 shuffle 轴表征程度 ---
fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
for ax, r2, label, color in zip(
    axes, [r2_a, r2_i], ['Ani shuffle', 'Inani shuffle'], [COLOR_ANI, COLOR_INANI],
):
    valid = np.isfinite(r2)
    ax.scatter(np.where(valid)[0], r2[valid], s=8, c=color, alpha=0.55, edgecolors='none', rasterized=True)
    med = np.nanmedian(r2)
    ax.axhline(med, color='k', ls='--', lw=1.0, label=f'median = {med:.3f}')
    ax.set_ylabel('R² (rsp ~ shuffle load)')
    ax.set_title(f'{area}  {label}')
    ax.legend(fontsize=8, loc='upper right')
axes[1].set_xlabel('Cell index')
fig.suptitle('Per-neuron shuffle-axis encoding', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_r2_scatter', area=area)

# --- fig B: 夹角分布 ---
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
for ax, ang, label, color in zip(
    axes, [ang_a, ang_i], ['Ani shuffle axis', 'Inani shuffle axis'], [COLOR_ANI, COLOR_INANI],
):
    valid = np.isfinite(ang)
    ax.hist(ang[valid], bins=30, color=color, alpha=0.55, edgecolor='white', lw=0.6)
    med = np.nanmedian(ang)
    ax.axvline(med, color='k', ls='--', lw=1.2, label=f'median = {med:.1f}°')
    ax.set_xlabel('Angle (°)  cell axis vs shuffle axis')
    ax.set_ylabel('Cell count')
    ax.set_title(label)
    ax.legend(fontsize=8)
fig.suptitle(f'{area}  axis alignment', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_angle_hist', area=area)

# --- fig C: R² 分布 ---
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
for ax, r2, label, color in zip(
    axes, [r2_a, r2_i], ['Ani', 'Inani'], [COLOR_ANI, COLOR_INANI],
):
    valid = r2[np.isfinite(r2)]
    ax.hist(valid, bins=30, color=color, alpha=0.55, edgecolor='white', lw=0.6)
    ax.axvline(np.median(valid), color='k', ls='--', lw=1.2,
               label=f'median = {np.median(valid):.3f}')
    ax.axvline(np.mean(valid), color='C1', ls=':', lw=1.2,
               label=f'mean = {np.mean(valid):.3f}')
    ax.set_xlabel('R² (rsp ~ shuffle load)')
    ax.set_ylabel('Cell count')
    ax.set_title(f'{label}  N = {n_cell}')
    ax.legend(fontsize=8)
fig.suptitle(f'{area}  shuffle-axis R² distribution', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_r2_hist', area=area)

# --- fig D: 夹角 vs R² ---
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
for ax, ang, r2, label, color in zip(
    axes,
    [ang_a, ang_i], [r2_a, r2_i], ['Ani', 'Inani'], [COLOR_ANI, COLOR_INANI],
):
    m = np.isfinite(ang) & np.isfinite(r2)
    ax.scatter(ang[m], r2[m], s=10, c=color, alpha=0.45, edgecolors='none', rasterized=True)
    ax.set_xlabel('Angle (°)')
    ax.set_ylabel('R² (rsp ~ shuffle load)')
    ax.set_title(label)
    ax.grid(True, ls=':', lw=0.5, alpha=0.4)
fig.suptitle(f'{area}  alignment vs encoding', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_alignment_vs_encoding', area=area)

r2_all_pop = dn['r2_shuf_all']
ang_all_pop = dn['angle_all']

fig, ax = plt.subplots(figsize=(8, 2.5))
valid = np.isfinite(r2_all_pop)
ax.scatter(np.where(valid)[0], r2_all_pop[valid], s=8, c=COLOR_ALL, alpha=0.55, edgecolors='none', rasterized=True)
med = np.nanmedian(r2_all_pop)
ax.axhline(med, color='k', ls='--', lw=1.0, label=f'median = {med:.3f}')
ax.set_xlabel('Cell index')
ax.set_ylabel('R² (rsp ~ shuffle load)')
ax.set_title(f'{area}  All shuffle (40 obj, N=1000)')
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_all_r2_scatter', area=area)

fig, ax = plt.subplots(figsize=(4, 3.2))
valid = np.isfinite(ang_all_pop)
ax.hist(ang_all_pop[valid], bins=30, color=COLOR_ALL, alpha=0.55, edgecolor='white', lw=0.6)
med = np.nanmedian(ang_all_pop)
ax.axvline(med, color='k', ls='--', lw=1.2, label=f'median = {med:.1f}°')
ax.set_xlabel('Angle (°)  cell axis vs shuffle axis')
ax.set_ylabel('Cell count')
ax.set_title('All shuffle axis')
ax.legend(fontsize=8)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_all_angle_hist', area=area)

fig, ax = plt.subplots(figsize=(4, 3.2))
valid = r2_all_pop[np.isfinite(r2_all_pop)]
ax.hist(valid, bins=30, color=COLOR_ALL, alpha=0.55, edgecolor='white', lw=0.6)
ax.axvline(np.median(valid), color='k', ls='--', lw=1.2, label=f'median = {np.median(valid):.3f}')
ax.axvline(np.mean(valid), color='C1', ls=':', lw=1.2, label=f'mean = {np.mean(valid):.3f}')
ax.set_xlabel('R² (rsp ~ shuffle load)')
ax.set_ylabel('Cell count')
ax.set_title(f'All  N = {n_cell}')
ax.legend(fontsize=8)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_all_r2_hist', area=area)

fig, ax = plt.subplots(figsize=(4, 3.5))
m = np.isfinite(ang_all_pop) & np.isfinite(r2_all_pop)
ax.scatter(ang_all_pop[m], r2_all_pop[m], s=10, c=COLOR_ALL, alpha=0.45, edgecolors='none', rasterized=True)
ax.set_xlabel('Angle (°)')
ax.set_ylabel('R² (rsp ~ shuffle load)')
ax.set_title('All')
ax.grid(True, ls=':', lw=0.5, alpha=0.4)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_all_alignment', area=area)

#%% 4demo.单细胞：shuffle 载荷 vs 响应（只读缓存，改 plot_cell 后重跑）
plot_cell = 850
n_extreme = 5

dn = np.load(resolve_area_path(savepath, check_area, 'shuffle_neuron'), allow_pickle=True)
rsp = np.load(rsp_path(cell_rootpath, check_area))
pc = int(plot_cell)
area = str(dn['check_area'])

def _demo_shuffle_cell(loads, obj_ids, mask, color, title_tag, r2_val, ang_val, color_by_ani=False):
    if color_by_ani:
        x_fit, y_fit = loads, rsp[pc]
    else:
        x_fit, y_fit = loads[mask], rsp[pc, mask]
    xl = np.linspace(x_fit.min(), x_fit.max(), 100)
    coef = np.polyfit(x_fit, y_fit, 1)

    # 与 plot3 一致：按 parent 的 S4 载荷排序；lo=Raw，hi=S4
    load_s4 = np.array([loads[o + 4 * 40] for o in obj_ids])
    order = np.argsort(load_s4)
    lo_parents = [obj_ids[i] for i in order[:n_extreme]]
    hi_parents = [obj_ids[i] for i in order[-n_extreme:][::-1]]
    lo_stim = [int(p) for p in lo_parents]           # Raw
    hi_stim = [int(p + 4 * 40) for p in hi_parents]  # S4

    fig = plt.figure(figsize=(7.5, 5.5))
    gs = fig.add_gridspec(n_extreme, 3, width_ratios=[1, 3.2, 1], hspace=0.12, wspace=0.08)
    ax = fig.add_subplot(gs[:, 1])
    axes_lo = [fig.add_subplot(gs[i, 0]) for i in range(n_extreme)]
    axes_hi = [fig.add_subplot(gs[i, 2]) for i in range(n_extreme)]

    if color_by_ani:
        ax.scatter(loads[mask_ani], rsp[pc, mask_ani], s=10, alpha=0.35, c=COLOR_ANI,
                   edgecolors='none', zorder=1)
        ax.scatter(loads[mask_inani], rsp[pc, mask_inani], s=10, alpha=0.35, c=COLOR_INANI,
                   edgecolors='none', zorder=1)
    else:
        ax.scatter(x_fit, y_fit, s=10, alpha=0.28, c='0.78', edgecolors='none', zorder=1)
    ax.plot(xl, np.polyval(coef, xl), color=color, lw=1.5, zorder=2)
    ax.scatter([loads[s] for s in lo_stim], [rsp[pc, s] for s in lo_stim],
               s=36, facecolors='none', edgecolors='0.35', lw=1.2, zorder=4)
    ax.scatter([loads[s] for s in hi_stim], [rsp[pc, s] for s in hi_stim],
               s=36, facecolors='none', edgecolors=color, lw=1.2, zorder=4)

    for i, stim in enumerate(lo_stim):
        axes_lo[i].imshow(Image.open(meta_paths[stim]))
        axes_lo[i].set_xticks([])
        axes_lo[i].set_yticks([])
        for sp in axes_lo[i].spines.values():
            sp.set_edgecolor('0.35')
            sp.set_linewidth(1.2)
        fig.add_artist(ConnectionPatch(
            (1.0, 0.5), (loads[stim], rsp[pc, stim]), 'axes fraction', 'data',
            axesA=axes_lo[i], axesB=ax, color='0.35', lw=0.7, alpha=0.55, zorder=3,
        ))
    for i, stim in enumerate(hi_stim):
        axes_hi[i].imshow(Image.open(meta_paths[stim]))
        axes_hi[i].set_xticks([])
        axes_hi[i].set_yticks([])
        for sp in axes_hi[i].spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(1.2)
        fig.add_artist(ConnectionPatch(
            (0.0, 0.5), (loads[stim], rsp[pc, stim]), 'axes fraction', 'data',
            axesA=axes_hi[i], axesB=ax, color=color, lw=0.7, alpha=0.55, zorder=3,
        ))

    ax.set_xlabel('Loading on shuffle axis')
    ax.set_ylabel('Response (spikes)')
    ax.set_title(f'{area} cell {pc}  {title_tag}  R²={r2_val:.3f}  angle={ang_val:.1f}°')
    fig.tight_layout()
    tag_slug = title_tag.lower().replace(' ', '_')
    finish_fig(fig, savepath, SCRIPT_THOUGHT, f'demo_cell{pc}_shuffle_{tag_slug}', area=area)


_demo_shuffle_cell(load_ani, ani_objs, mask_ani, COLOR_ANI, 'Ani shuffle',
                   float(dn['r2_shuf_ani'][pc]), float(dn['angle_ani'][pc]))
_demo_shuffle_cell(load_inani, inani_objs, mask_inani, COLOR_INANI, 'Inani shuffle',
                   float(dn['r2_shuf_inani'][pc]), float(dn['angle_inani'][pc]))
_demo_shuffle_cell(load_all, all_objs, mask_all, COLOR_ALL, 'All shuffle',
                   float(dn['r2_shuf_all'][pc]), float(dn['angle_all'][pc]),
                   color_by_ani=True)

#%%

