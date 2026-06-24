'''
中介验证：metamer 打乱对发放率的影响是否完全由 prefer axis loading 变化所中介。

核心逻辑：shuffle → Δ object-space coords → Δload on prefer axis → Δfiring
若 load 已完全解释 shuffle 的效果，则控制 load 后 shuffle 无额外解释力（ΔR² ≈ 0）。

依赖缓存（只读）：
  CACHE3       : {area}_obj_axis_fit.npz    — meta_load, axes, r2, F_mu/std
  CACHE_NEURON : {area}_shuffle_neuron.npz  — angle_ani/inani, r2_shuf_*
  avr_rsp.npy  : 原始神经响应
'''

#%% 0. 配置 & 加载缓存
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import OS_Tools as ot

# --- 路径 ---
savepath      = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis'
cell_rootpath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
check_area    = 'ASB'

N_METAMER = 1000
N_OBJ     = 40
N_SHUF    = 5
COLOR_ANI   = '#c0392b'
COLOR_INANI = '#2980b9'
SHUF_LABELS = ['Raw', 'S1', 'S2', 'S3', 'S4']

from obj_space_paths import area_path, resolve_area_path, rsp_path
from obj_space_plot import finish_fig, SCRIPT_MEDIATION

SAVE_FIGURES = True
SHOW_FIGURES = True

# --- 缓存路径 ---
CACHE3        = resolve_area_path(savepath, check_area, 'obj_axis_fit')
CACHE_NEURON  = resolve_area_path(savepath, check_area, 'shuffle_neuron')
CACHE_MEDIATION = resolve_area_path(savepath, check_area, 'mediation')
RSP_PATH      = rsp_path(cell_rootpath, check_area)

# --- 加载 ---
d3  = np.load(CACHE3,       allow_pickle=True)
dn  = np.load(CACHE_NEURON, allow_pickle=True)
rsp = np.load(RSP_PATH)                          # (n_cell, 1000)

meta_load = d3['meta_load'].astype(np.float32)   # (n_cell, 1000)
r2_load   = d3['r2'].astype(np.float32)          # (n_cell,)   R²(rsp ~ load)
axes_cell = d3['axes'].astype(np.float32)        # (n_cell, 50)

angle_ani   = dn['angle_ani'].astype(np.float32)    # (n_cell,)
angle_inani = dn['angle_inani'].astype(np.float32)
r2_shuf_ani   = dn['r2_shuf_ani'].astype(np.float32)
r2_shuf_inani = dn['r2_shuf_inani'].astype(np.float32)

n_cell = rsp.shape[0]
assert meta_load.shape == (n_cell, N_METAMER)

# --- stimulus labels ---
idx      = np.arange(N_METAMER)
within   = idx % 200
shuffle  = within // 40           # 0=Raw … 4=S4
is_ani   = (within % 40) < 20
parent   = within % 40            # 0–39  (0–19 ani, 20–39 inani)

print(f'{check_area}: {n_cell} cells loaded')
print(f'r2_load   median={np.nanmedian(r2_load):.3f}')

#%% 1. 构建 avg_load / avg_rsp 矩阵，计算斜率
'''
avg_load[i, o, s] = mean over 5 cycles of meta_load[i, j]  where parent==o, shuffle==s
avg_rsp [i, o, s] = mean over 5 cycles of rsp[i, j]         where parent==o, shuffle==s

slope_load[i, o] = linear slope of avg_load[i, o, :] ~ shuffle (0–4)
slope_rsp [i, o] = linear slope of avg_rsp [i, o, :] ~ shuffle (0–4)

向量化：用 index-mapping 一次完成，无三重循环
'''

# 为每个 (object, shuffle) 组合建立 index 列表（1000 个 j 分成 200 组，每组 5 个）
# group_idx[o, s] = list of j in that (object=o, shuffle=s) bin
group_idx = np.full((N_OBJ, N_SHUF, N_METAMER // (N_OBJ * N_SHUF)), -1, dtype=int)
for o in range(N_OBJ):
    for s in range(N_SHUF):
        hits = np.where((parent == o) & (shuffle == s))[0]
        group_idx[o, s, :len(hits)] = hits

# (n_cell, N_OBJ, N_SHUF): 按 5 cycles 平均
# reshape trick: 把 1000 个 j 重排成 (N_OBJ, N_SHUF, 5) 再 mean
# group_idx (40, 5, 5) — last dim is 5 cycles
gi = group_idx.reshape(N_OBJ * N_SHUF, -1)          # (200, 5)
avg_load = meta_load[:, gi].mean(-1).reshape(n_cell, N_OBJ, N_SHUF)  # (n_cell, 40, 5)
avg_rsp  = rsp[:, gi].mean(-1).reshape(n_cell, N_OBJ, N_SHUF)        # (n_cell, 40, 5)

# 线性斜率：slope = Σ (x_c * y) / Σ x_c²,  x = [0..4], x_c = x - 2,  Σ x_c² = 10
shuf_c = np.arange(N_SHUF, dtype=np.float32) - 2.0    # [-2,-1,0,1,2]
slope_load = (avg_load * shuf_c).sum(-1) / 10.0       # (n_cell, 40)
slope_rsp  = (avg_rsp  * shuf_c).sum(-1) / 10.0       # (n_cell, 40)

print(f'avg_load shape: {avg_load.shape}')
print(f'slope_load range: [{slope_load.min():.3f}, {slope_load.max():.3f}]')
print(f'slope_rsp  range: [{slope_rsp.min():.3f}, {slope_rsp.max():.3f}]')

#%% 2. Test 1 — 斜率散点 & 每神经元 Pearson r
'''
假设验证：slope_rsp ∝ slope_load（scatter 紧密线性）
每神经元 Pearson r(slope_load, slope_rsp) over 40 objects → 分布集中于高正值
'''
rsp_unit = rsp_unit if 'rsp_unit' in dir() else 'spikes'   # 单独运行时的 fallback

# --- 每神经元 Pearson r ---
# slope_load (n_cell, 40), slope_rsp (n_cell, 40)
sl_mu = slope_load.mean(1, keepdims=True)
sr_mu = slope_rsp.mean(1, keepdims=True)
sl_c  = slope_load - sl_mu
sr_c  = slope_rsp  - sr_mu
cov   = (sl_c * sr_c).sum(1)
std_l = np.sqrt((sl_c ** 2).sum(1))
std_r = np.sqrt((sr_c ** 2).sum(1))
denom = std_l * std_r
pearson_r = np.where(denom > 1e-8, cov / denom, np.nan)   # (n_cell,)

print(f'Per-neuron Pearson r — median={np.nanmedian(pearson_r):.3f}  '
      f'mean={np.nanmean(pearson_r):.3f}  '
      f'frac>0.5: {np.nanmean(pearson_r > 0.5):.2%}')

# --- 全局散点（subsample if huge: n_cell × 40 可达 40k 点）---
from scipy import stats as sp_stats

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# 左：全局散点，颜色 = ani/inani
is_ani_obj = np.arange(N_OBJ) < 20          # (40,)  broadcast over cells
sl_flat = slope_load.ravel()               # (n_cell × 40,)
sr_flat = slope_rsp.ravel()
ani_flat = np.tile(is_ani_obj, n_cell)     # (n_cell × 40,)

ax = axes[0]
ax.scatter(sl_flat[ani_flat],  sr_flat[ani_flat],  s=4, alpha=0.15,
           c=COLOR_ANI,   edgecolors='none', rasterized=True, label='Ani')
ax.scatter(sl_flat[~ani_flat], sr_flat[~ani_flat], s=4, alpha=0.15,
           c=COLOR_INANI, edgecolors='none', rasterized=True, label='Inani')

# 全局线性拟合（scipy 提供完整统计）
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
      f'intercept={lr.intercept:.5f}  R²={lr.rvalue**2:.4f}  p={lr.pvalue:.2e}')

# 右：每神经元 Pearson r 分布
ax = axes[1]
valid_r = pearson_r[np.isfinite(pearson_r)]
ax.hist(valid_r, bins=40, color='0.45', edgecolor='white', lw=0.5)
ax.axvline(np.nanmedian(pearson_r), color='C1', ls='--', lw=1.5,
           label=f'median = {np.nanmedian(pearson_r):.3f}')
ax.axvline(0, color='k', ls=':', lw=1.0)
ax.set_xlabel('Pearson r  (slope_load vs slope_rsp,  40 objects)')
ax.set_ylabel('Cell count')
ax.set_title(f'{check_area}  per-neuron r  N={len(valid_r)}')
ax.legend(fontsize=8)

fig.suptitle('Test 1: Shuffle slope correlation — load drives rsp', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_MEDIATION, 'test1_slope_scatter', area=check_area)

#%% 3. Test 2 — 增量 ΔR² 检验
'''
R²_load : R²(rsp ~ load)           已有
R²_full : R²(rsp ~ load + shuffle) 新计算
ΔR²     = R²_full - R²_load        → shuffle 控制 load 后的"直接"贡献

若 ΔR² ≈ 0 → 样本空间模型几乎完全解释 shuffle 效应
'''

# --- 计算 R²_full per neuron ---
shuf_f = shuffle.astype(np.float32)
shuf_z = (shuf_f - shuf_f.mean()) / (shuf_f.std() + 1e-8)   # z-score shuffle

r2_full = np.full(n_cell, np.nan, np.float32)
for i in range(n_cell):
    X = np.c_[meta_load[i], shuf_z, np.ones(N_METAMER)]
    coef, _, _, _ = np.linalg.lstsq(X, rsp[i], rcond=None)
    pred = X @ coef
    ss_res = ((rsp[i] - pred) ** 2).sum()
    ss_tot = rsp[i].var() * N_METAMER
    r2_full[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

delta_r2 = r2_full - r2_load    # (n_cell,)

valid = np.isfinite(delta_r2)
print(f'ΔR² = R²_full - R²_load:')
print(f'  median = {np.nanmedian(delta_r2):.4f}')
print(f'  mean   = {np.nanmean(delta_r2):.4f}')
print(f'  p95    = {np.nanpercentile(delta_r2, 95):.4f}')
print(f'  frac < 0.01: {np.nanmean(delta_r2 < 0.01):.2%}')

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

# 图 A：ΔR² 分布
ax = axes[0]
dr = delta_r2[valid]
ax.hist(dr, bins=40, color='0.45', edgecolor='white', lw=0.5)
ax.axvline(np.median(dr), color='C1', ls='--', lw=1.5,
           label=f'median={np.median(dr):.4f}')
ax.axvline(0, color='k', ls=':', lw=1.0)
ax.set_xlabel('ΔR²  (R²_full − R²_load)')
ax.set_ylabel('Cell count')
ax.set_title('A: ΔR² distribution')
ax.legend(fontsize=8)

# 图 B：scatter R²_load vs ΔR²
ax = axes[1]
m = valid & np.isfinite(r2_load)
ax.scatter(r2_load[m], delta_r2[m], s=8, alpha=0.4, c='0.4',
           edgecolors='none', rasterized=True)
ax.axhline(0, color='C1', ls='--', lw=1.0)
ax.set_xlabel('R²_load  (object space model)')
ax.set_ylabel('ΔR²  (additional from shuffle)')
ax.set_title('B: R²_load vs ΔR²')
ax.grid(True, ls=':', lw=0.5, alpha=0.4)

fig.suptitle(f'{check_area}  Test 2: Incremental ΔR² — does shuffle add beyond load?',
             fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_MEDIATION, 'test2_delta_r2', area=check_area)

#%% 4. Demo — 单神经元 3 图面板
'''
图 A: load vs rsp 散点（基准对象空间模型）
图 B: per-object 轨迹 — 上: avg_load ~ shuffle,  下: avg_rsp ~ shuffle
图 C: 模型残差 vs shuffle level（应近似水平）

选取神经元：高 R²_load 且 ΔR² 低，清楚展示中介效应
'''
demo_cell = int(np.nanargmax(r2_load))   # 默认取 R²_load 最高的神经元；可手动覆盖
# demo_cell = 300

n_obj_show = 4     # per-object 轨迹图中展示多少个 object（分 ani / inani 各半）

pc = demo_cell
print(f'Demo cell: {pc}  R²_load={r2_load[pc]:.3f}  '
      f'ΔR²={delta_r2[pc]:.4f}  angle_ani={angle_ani[pc]:.1f}°  angle_inani={angle_inani[pc]:.1f}°')

x = meta_load[pc]    # (1000,) loading on prefer axis
y = rsp[pc]          # (1000,) firing rate
m_fit, b_fit = np.polyfit(x, y, 1)
resid = y - (m_fit * x + b_fit)

# --- 图 A: load vs rsp ---
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(x, y, s=8, alpha=0.28, c='0.75', edgecolors='none', zorder=1)
xl = np.linspace(x.min(), x.max(), 100)
ax.plot(xl, m_fit * xl + b_fit, 'r-', lw=2, zorder=3)
ax.set_xlabel('Loading on preferred axis')
ax.set_ylabel(f'Response ({rsp_unit})')
ax.set_title(f'{check_area} cell {pc}  A: Load → Rsp  R²={r2_load[pc]:.3f}')
ax.grid(True, ls=':', lw=0.5, alpha=0.4)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_MEDIATION, f'demo_cell{pc}_panel_A', area=check_area)

# --- 图 B: per-object 轨迹（选 ani top 4 + inani top 4 by |slope_rsp|）---
half = n_obj_show // 2
ani_objs  = np.arange(20)
inani_objs = np.arange(20, 40)
sel_ani   = ani_objs  [np.argsort(np.abs(slope_rsp[pc, ani_objs ]))[-half:]]
sel_inani = inani_objs[np.argsort(np.abs(slope_rsp[pc, inani_objs]))[-half:]]
sel_objs  = np.concatenate([sel_ani, sel_inani])

cmap_ani   = plt.cm.Reds
cmap_inani = plt.cm.Blues
shuf_x = np.arange(N_SHUF)

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
for k, o in enumerate(sel_ani):
    c = cmap_ani(0.35 + 0.55 * k / (half - 1 + 1e-8))
    ax_top.plot(shuf_x, avg_load[pc, o, :], '-o', color=c, ms=4, lw=1.4, alpha=0.85)
    ax_bot.plot(shuf_x, avg_rsp [pc, o, :], '-o', color=c, ms=4, lw=1.4, alpha=0.85)
for k, o in enumerate(sel_inani):
    c = cmap_inani(0.35 + 0.55 * k / (half - 1 + 1e-8))
    ax_top.plot(shuf_x, avg_load[pc, o, :], '--s', color=c, ms=4, lw=1.4, alpha=0.85)
    ax_bot.plot(shuf_x, avg_rsp [pc, o, :], '--s', color=c, ms=4, lw=1.4, alpha=0.85)

ax_top.set_ylabel('Avg load on prefer axis')
ax_top.set_title(f'{check_area} cell {pc}  B: Per-object trajectories across shuffle')
ax_top.grid(True, ls=':', lw=0.5, alpha=0.4)
ax_bot.set_ylabel('Avg firing rate (spikes)')
ax_bot.set_xlabel('Shuffle level')
ax_bot.set_xticks(shuf_x)
ax_bot.set_xticklabels(SHUF_LABELS)
ax_bot.grid(True, ls=':', lw=0.5, alpha=0.4)

leg = [Line2D([0],[0], color='0.6', ls='-',  marker='o', ms=5, label='Ani'),
       Line2D([0],[0], color='0.6', ls='--', marker='s', ms=5, label='Inani')]
ax_top.legend(handles=leg, fontsize=8, loc='best')
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_MEDIATION, f'demo_cell{pc}_panel_B', area=check_area)

# --- 图 C: 残差 vs shuffle ---
fig, ax = plt.subplots(figsize=(5, 3.8))
ax.scatter(shuffle + np.random.randn(N_METAMER) * 0.07,
           resid, s=6, alpha=0.18, c='0.55', edgecolors='none', rasterized=True)

# 残差 vs shuffle 回归
for flag, color, label in ((True, COLOR_ANI, 'Ani'), (False, COLOR_INANI, 'Inani')):
    m_r, b_r = np.polyfit(shuffle[is_ani == flag], resid[is_ani == flag], 1)
    xl = np.array([0, 4], dtype=float)
    ax.plot(xl, m_r * xl + b_r, '-', color=color, lw=2, label=f'{label} slope={m_r:.3f}')

# 每个 shuffle bin 的均值 ± SE
for s in range(N_SHUF):
    m_s = shuffle == s
    mu, se = resid[m_s].mean(), resid[m_s].std() / np.sqrt(m_s.sum())
    ax.errorbar(s, mu, yerr=se, fmt='D', color='k', ms=6, capsize=4, zorder=5)

ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Shuffle level')
ax.set_ylabel('Residual  (rsp − load model)')
ax.set_xticks(np.arange(N_SHUF))
ax.set_xticklabels(SHUF_LABELS)
ax.set_title(f'{check_area} cell {pc}  C: Residual vs Shuffle  (after load model)')
ax.legend(fontsize=8)
ax.grid(True, ls=':', lw=0.5, alpha=0.4)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_MEDIATION, f'demo_cell{pc}_panel_C', area=check_area)

#%% 6a. Demo 筛选 — ani 和 inani 分别排名（只需跑一次）
'''
对 ani object 和 inani object 分别独立筛选最佳神经元：
  ranked_ani   — 最能体现 ani-obj 的 load~rsp 耦合的神经元
  ranked_inani — 最能体现 inani-obj 的 load~rsp 耦合的神经元
群体统计图也在此 cell 输出。
'''
# ── 参数 ──────────────────────────────────────────────────────────────────────
demo_ani_obj   = 17    # ani   object 0-indexed (0–19)，对应"图9"
demo_inani_obj = 26   # inani object 0-indexed (20–39)，对应"图31"

# slope_sign:  +1 = 反例（打乱↑ → load↑ → rsp↑）
#              -1 = 正例（打乱↑ → load↓ → rsp↓）
#               0 = 不限方向
slope_sign  = +1
top_n_print = 10
# ─────────────────────────────────────────────────────────────────────────────

# 每个 (neuron, object) 的 load ~ rsp 相关（across 5 shuffle levels）
al_c = avg_load - avg_load.mean(-1, keepdims=True)
ar_c = avg_rsp  - avg_rsp.mean(-1, keepdims=True)
cor_per_obj = np.clip(
    (al_c * ar_c).sum(-1) / (
        np.sqrt((al_c**2).sum(-1)) * np.sqrt((ar_c**2).sum(-1)) + 1e-16
    ), -1, 1
)    # (n_cell, 40)

dir_label6 = ('反例 (↑load↑rsp)' if slope_sign > 0 else
               '正例 (↓load↓rsp)' if slope_sign < 0 else '不限方向')


def _rank_for_obj(obj_idx, sign):
    """按单个 object 的 cor×R²_load 排序，加方向约束。"""
    sc = cor_per_obj[:, obj_idx] * r2_load
    sc = sc.copy().astype(float)
    sc[~np.isfinite(sc)] = np.nan
    if sign != 0:
        sc[slope_load[:, obj_idx] * sign <= 0] = np.nan
    return np.argsort(np.nan_to_num(sc, nan=-np.inf))[::-1], sc


ranked_ani,   score_ani   = _rank_for_obj(demo_ani_obj,   slope_sign)
ranked_inani, score_inani = _rank_for_obj(demo_inani_obj, slope_sign)

# 打印候选表
for tag, ranked, score, obj_idx in [
    ('ANI',   ranked_ani,   score_ani,   demo_ani_obj),
    ('INANI', ranked_inani, score_inani, demo_inani_obj),
]:
    print(f'\n── {tag} obj {obj_idx+1}  [{dir_label6}] ──')
    print(f'{"rank":>4}  {"cell":>6}  {"score":>7}  {"R²_load":>8}  '
          f'{"cor":>7}  {"slope_load":>11}  {"slope_rsp":>10}')
    for rank, ci in enumerate(ranked[:top_n_print], 1):
        if np.isnan(score[ci]):
            break
        print(f'{rank:>4}  {ci:>6}  {score[ci]:>7.4f}  {r2_load[ci]:>8.3f}  '
              f'{cor_per_obj[ci, obj_idx]:>7.3f}  '
              f'{slope_load[ci, obj_idx]:>11.4f}  '
              f'{slope_rsp [ci, obj_idx]:>10.4f}')

# 群体统计图（不变）
cor_ani_flat   = cor_per_obj[:, :20].ravel()
cor_inani_flat = cor_per_obj[:, 20:].ravel()
cor_ani_flat   = cor_ani_flat  [np.isfinite(cor_ani_flat)]
cor_inani_flat = cor_inani_flat[np.isfinite(cor_inani_flat)]

fig, ax = plt.subplots(figsize=(5.5, 3.8))
bins = np.linspace(-1, 1, 35)
ax.hist(cor_ani_flat,   bins=bins, color=COLOR_ANI,   alpha=0.55, edgecolor='white',
        lw=0.5, label=f'Ani  (median={np.median(cor_ani_flat):.3f})')
ax.hist(cor_inani_flat, bins=bins, color=COLOR_INANI, alpha=0.55, edgecolor='white',
        lw=0.5, label=f'Inani (median={np.median(cor_inani_flat):.3f})')
ax.axvline(np.median(cor_ani_flat),   color=COLOR_ANI,   ls='--', lw=1.5)
ax.axvline(np.median(cor_inani_flat), color=COLOR_INANI, ls='--', lw=1.5)
ax.axvline(0, color='k', ls=':', lw=0.8)
ax.set_xlabel('Pearson r  (load ~ rsp,  5 shuffle points,  per neuron × object)')
ax.set_ylabel('Count  (neuron × object pairs)')
ax.set_title(f'{check_area}  Load–Rsp coupling: Ani vs Inani objects')
ax.legend(fontsize=9)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_MEDIATION, 'test6a_cor_coupling', area=check_area)

#%% 6b. Demo 单细胞图 — 两张图分别对应 ani / inani 最佳神经元
# ── 参数（改完重跑此 cell）────────────────────────────────────────────────────
rank_ani   = 2      # ani   排名（1=最强）；指定 cell_ani   则忽略
rank_inani = 5     # inani 排名；指定 cell_inani 则忽略
cell_ani   = None   # 手动指定 ani   神经元 index
cell_inani = None   # 手动指定 inani 神经元 index
# ─────────────────────────────────────────────────────────────────────────────

pc_ani   = int(cell_ani)   if cell_ani   is not None else int(ranked_ani  [rank_ani   - 1])
pc_inani = int(cell_inani) if cell_inani is not None else int(ranked_inani[rank_inani - 1])

shuf_x = np.arange(N_SHUF)


def _single_cell_plot(pc, obj_idx, color, tag):
    img_id = obj_idx + 1    # 1-indexed 图片编号
    print(f'{tag} — cell {pc}  img {img_id}  R²={r2_load[pc]:.3f}  '
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
        f'{check_area} cell {pc}  img {img_id}  R²={r2_load[pc]:.3f} ',
        fontsize=10
    )
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_MEDIATION, f'demo_{tag.lower()}_cell{pc}_obj{img_id}', area=check_area)


_single_cell_plot(pc_ani,   demo_ani_obj,   COLOR_ANI,   'Ani')
_single_cell_plot(pc_inani, demo_inani_obj, COLOR_INANI, 'Inani')

#%%
