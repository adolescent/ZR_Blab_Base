"""Generate and save all object-space analysis figures from cached npz files."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import matplotlib.colors as mcolors
from PIL import Image
from scipy import stats as sp_stats

from obj_space_paths import BRAIN_AREAS, resolve_area_path, rsp_path, shared_path
from obj_space_plot import (
    SCRIPT_MEDIATION,
    SCRIPT_TEST_RSP,
    SCRIPT_THOUGHT,
    configure,
    finish_fig,
)

N_DIM = 50
N_METAMER = 1000
N_OBJ = 40
N_SHUF = 5
COLOR_ANI = '#c0392b'
COLOR_INANI = '#2980b9'
COLOR_ALL = '#2c3e50'
SHUF_LABELS = ['Raw', 'S1', 'S2', 'S3', 'S4']
SHUF_ALPHA = [1.0, 0.72, 0.52, 0.34, 0.18]
SHUF_ALPHA_DEMO = [1.0, 0.62, 0.38, 0.20, 0.07]
SHUF_SAT = [0.22, 0.42, 0.62, 0.82, 1.0]


def _stim_labels(n_metamer=N_METAMER):
    idx = np.arange(n_metamer)
    within = idx % 200
    shuffle = within // 40
    is_ani = (within % 40) < 20
    parent_id = within % 40
    plot_idx = idx < 200
    return idx, within, shuffle, is_ani, parent_id, plot_idx


def _rainbow_parent_colors(n_obj):
    return [plt.cm.rainbow(t)[:3] for t in np.linspace(0, 1, n_obj)]


def _color_by_shuffle(parent_rgb, shuffle_level, sat_levels=SHUF_SAT):
    h, _, v = mcolors.rgb_to_hsv(parent_rgb)
    return mcolors.hsv_to_rgb([h, sat_levels[shuffle_level], v])


def _extreme(scores, k=10):
    order = np.argsort(scores)
    return order[-k:][::-1], order[:k]


def _fit_shuffle_axis(meta_coords, obj_ids, n_dim=N_DIM):
    feats, ys = [], []
    per_obj_raw = {}
    for obj in obj_ids:
        raw = meta_coords[obj, :n_dim]
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
    return w.astype(np.float32), float(r2_group), per_obj_raw


def _loadings(meta_coords, w, per_obj_raw, obj_ids, n_dim=N_DIM):
    loads = np.zeros(N_METAMER, np.float32)
    for obj in obj_ids:
        for s in range(5):
            i = obj + s * 40
            loads[i] = (meta_coords[i, :n_dim] - per_obj_raw[obj]) @ w
    return loads


def _per_obj_r2(loads, obj_ids):
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


def plot_test_rsp_shared(savepath, plot_pc=3, n_extreme=10):
    d1 = np.load(shared_path(savepath, 'step1'), allow_pickle=True)
    d2 = np.load(shared_path(savepath, 'step2'), allow_pickle=True)
    coords = d1['coords']
    cumvar = d1['cumvar']
    ev_ratio = d1['ev_ratio'] if 'ev_ratio' in d1.files else np.diff(np.concatenate([[0], cumvar]))[:N_DIM]
    img_paths = list(d1['img_paths'])
    nsd_coords = d1['coords']
    meta_coords = d2['coords']

    plot_pc = int(plot_pc)
    pc_idx = plot_pc - 1
    scores = coords[:, pc_idx]
    hi_idx, lo_idx = _extreme(scores, n_extreme)

    fig, axes = plt.subplots(2, n_extreme, figsize=(1.2 * n_extreme, 2.8))
    for col, i in enumerate(hi_idx):
        axes[0, col].imshow(Image.open(img_paths[i]))
        axes[0, col].axis('off')
    for col, i in enumerate(lo_idx):
        axes[1, col].imshow(Image.open(img_paths[i]))
        axes[1, col].axis('off')
    axes[0, 0].set_ylabel(f'PC{plot_pc} hi', fontsize=9)
    axes[1, 0].set_ylabel(f'PC{plot_pc} lo', fontsize=9)
    fig.suptitle(f'NSD1k extremes on PC{plot_pc} ({ev_ratio[pc_idx]:.1%} var)', fontsize=10)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_TEST_RSP, f'step1_pc{plot_pc}_extremes')

    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, 'k-', lw=1.5)
    ax.axvline(N_DIM, color='C1', ls='--', lw=0.8, label=f'K={N_DIM} ({cumvar[N_DIM - 1]:.1%})')
    ax.set_xlim(1, 100)
    ax.set_xlabel('N PCs')
    ax.set_ylabel('Explained VAR')
    ax.legend(fontsize=8)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'step1_pca_variance')

    idx, within, shuffle, is_ani, _, plot_idx = _stim_labels()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(nsd_coords[:, 0], nsd_coords[:, 1], s=6, c='#dddddd', edgecolors='none',
               label='NSD1k', zorder=1, rasterized=True)
    for shuf, alpha in zip(range(5), SHUF_ALPHA):
        for ani_flag, color in ((True, COLOR_ANI), (False, COLOR_INANI)):
            m = plot_idx & (shuffle == shuf) & (is_ani == ani_flag)
            ax.scatter(meta_coords[m, 0], meta_coords[m, 1], s=22, c=color, alpha=alpha,
                       edgecolors='white', linewidths=0.25, zorder=3 + shuf, rasterized=True)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('NSD1k object space (PC1–PC2)')
    ax.grid(True, ls=':', lw=0.6, alpha=0.5)
    shuf_lev = np.arange(5, dtype=float)
    for obj in range(40):
        pts = meta_coords[[obj + s * 40 for s in range(5)], :2]
        c = COLOR_ANI if obj < 20 else COLOR_INANI
        m1, m2 = np.polyfit(shuf_lev, pts[:, 0], 1), np.polyfit(shuf_lev, pts[:, 1], 1)
        org = np.array([m1[1], m2[1]])
        vec = np.array([np.polyval(m1, 4) - org[0], np.polyval(m2, 4) - org[1]])
        ax.quiver(org[0], org[1], vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                  color=c, alpha=0.85, width=0.003, headwidth=4, headlength=5, zorder=6)
    leg_ani = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ANI, markersize=7, label='Ani'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INANI, markersize=7, label='Inani'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', markersize=6, label='NSD1k'),
        Line2D([0], [0], color='#555555', lw=1.3, alpha=0.85, label='Fitted vector (Raw→S4)'),
    ]
    leg_shuf = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', markersize=7,
               alpha=a, label=lab)
        for lab, a in zip(SHUF_LABELS, SHUF_ALPHA)
    ]
    leg1 = ax.legend(handles=leg_ani, loc='upper left', fontsize=8, framealpha=0.92, title='Category')
    ax.add_artist(leg1)
    ax.legend(handles=leg_shuf, loc='lower right', fontsize=8, framealpha=0.92, title='Shuffle')
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'step2_metamer_pc12')


def plot_test_rsp_area(savepath, cell_rootpath, area, plot_cell=780, demo_cell=300,
                       n_extreme=10, n_meta_plot=5, highlight_ids=None, highlight_range=None,
                       n_bins=40, sort_cells='r2', norm_mode='zscore', cell_norm_pct=(5, 95),
                       show_cells=None):
    if not os.path.isfile(resolve_area_path(savepath, area, 'obj_axis_fit')):
        print(f'[{area}] skip Test_Obj_Space_Rsp figures — missing obj_axis_fit')
        return

    d1 = np.load(shared_path(savepath, 'step1'), allow_pickle=True)
    d2 = np.load(shared_path(savepath, 'step2'), allow_pickle=True)
    d3 = np.load(resolve_area_path(savepath, area, 'obj_axis_fit'), allow_pickle=True)
    nsd_img_paths = list(d1['img_paths'])
    meta_paths = list(d2['img_paths'])
    nsd_coords = d1['coords']
    F_mu, F_std = d3['F_mu'], d3['F_std']
    axes_fit = d3['axes']
    r2 = d3['r2']
    meta_load = d3['meta_load']
    rsp = np.load(rsp_path(cell_rootpath, area))

    pc = int(plot_cell)
    nsd_load = ((nsd_coords - F_mu) / F_std) @ axes_fit[pc]
    order = np.argsort(nsd_load)
    hi_idx, lo_idx = order[-n_extreme:][::-1], order[:n_extreme]

    fig, axes_img = plt.subplots(2, n_extreme, figsize=(1.2 * n_extreme, 2.8))
    for col, j in enumerate(hi_idx):
        axes_img[0, col].imshow(Image.open(nsd_img_paths[j]))
        axes_img[0, col].axis('off')
    for col, j in enumerate(lo_idx):
        axes_img[1, col].imshow(Image.open(nsd_img_paths[j]))
        axes_img[1, col].axis('off')
    axes_img[0, 0].set_ylabel('axis hi', fontsize=9)
    axes_img[1, 0].set_ylabel('axis lo', fontsize=9)
    fig.suptitle(f'{area} cell {pc} — NSD extremes (R²={r2[pc]:.3f})', fontsize=10)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_TEST_RSP, f'cell{pc}_nsd_extremes', area=area)

    x = meta_load[pc]
    y = rsp[pc]
    xl = np.linspace(x.min(), x.max(), 100)
    m, b = np.polyfit(x, y, 1)
    load_u = meta_load[pc].reshape(5, 200).mean(0)
    hi_stim = np.argsort(load_u)[-n_meta_plot:][::-1]
    lo_stim = np.argsort(load_u)[:n_meta_plot]

    fig = plt.figure(figsize=(7.5, 5.5))
    gs = fig.add_gridspec(n_meta_plot, 3, width_ratios=[1, 3.2, 1], hspace=0.12, wspace=0.08)
    ax = fig.add_subplot(gs[:, 1])
    axes_lo = [fig.add_subplot(gs[i, 0]) for i in range(n_meta_plot)]
    axes_hi = [fig.add_subplot(gs[i, 2]) for i in range(n_meta_plot)]
    ax.scatter(x, y, s=10, alpha=0.28, c='0.78', edgecolors='none', zorder=1)
    ax.plot(xl, m * xl + b, 'r-', lw=1.5, zorder=2)
    ax.scatter(x[hi_stim], y[hi_stim], s=36, facecolors='none', edgecolors=COLOR_ANI, lw=1.2, zorder=4)
    ax.scatter(x[lo_stim], y[lo_stim], s=36, facecolors='none', edgecolors=COLOR_INANI, lw=1.2, zorder=4)
    for i, stim in enumerate(lo_stim):
        axes_lo[i].imshow(Image.open(meta_paths[stim]))
        axes_lo[i].set_xticks([])
        axes_lo[i].set_yticks([])
        for sp in axes_lo[i].spines.values():
            sp.set_edgecolor(COLOR_INANI)
            sp.set_linewidth(1.2)
        fig.add_artist(ConnectionPatch(
            (1.0, 0.5), (x[stim], y[stim]), 'axes fraction', 'data',
            axesA=axes_lo[i], axesB=ax, color=COLOR_INANI, lw=0.7, alpha=0.55, zorder=3,
        ))
    for i, stim in enumerate(hi_stim):
        axes_hi[i].imshow(Image.open(meta_paths[stim]))
        axes_hi[i].set_xticks([])
        axes_hi[i].set_yticks([])
        for sp in axes_hi[i].spines.values():
            sp.set_edgecolor(COLOR_ANI)
            sp.set_linewidth(1.2)
        fig.add_artist(ConnectionPatch(
            (0.0, 0.5), (x[stim], y[stim]), 'axes fraction', 'data',
            axesA=axes_hi[i], axesB=ax, color=COLOR_ANI, lw=0.7, alpha=0.55, zorder=3,
        ))
    ax.set_xlabel('Loading on preferred axis')
    ax.set_ylabel('Response (spikes)')
    ax.set_title(f'{area} cell {pc}  R²={r2[pc]:.3f}  (unique metamer hi/lo ×{n_meta_plot})')
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_TEST_RSP, f'cell{pc}_metamer_tuning', area=area)

    r2_pop = d3['r2']
    axes_pop = d3['axes']
    n_pop = len(r2_pop)
    r2_valid = r2_pop[np.isfinite(r2_pop)]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.hist(r2_valid, bins=30, color='0.55', edgecolor='white', lw=0.6)
    med = np.median(r2_valid)
    ax.axvline(med, color='C1', ls='--', lw=1.5, label=f'median = {med:.3f}')
    ax.axvline(np.mean(r2_valid), color='C0', ls=':', lw=1.2, label=f'mean = {np.mean(r2_valid):.3f}')
    ax.set_xlabel('R² (50D axis model)')
    ax.set_ylabel('Cell count')
    ax.set_title(f'{area}  N = {n_pop}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'pop_r2_hist', area=area)

    u = axes_pop / (np.linalg.norm(axes_pop, axis=1, keepdims=True) + 1e-8)
    cos_mat = np.clip(u @ u.T, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_mat))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(angle_deg, cmap='viridis', vmin=0, vmax=90, origin='lower', aspect='equal')
    ax.set_xlabel('Cell index')
    ax.set_ylabel('Cell index')
    ax.set_title(f'{area}  pairwise axis angle ({n_pop}×{n_pop})')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Angle (°)')
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'pop_axis_angle_matrix', area=area)

    pc = int(demo_cell)
    x = d3['meta_load'][pc]
    y = rsp[pc]
    r2_demo = float(d3['r2'][pc])
    idx, _, shuffle, is_ani, parent_id, _ = _stim_labels()
    parent_id = parent_id + 1
    hl_set = set()
    if highlight_ids is not None:
        hl_set.update(highlight_ids)
    if highlight_range is not None:
        hl_set.update(range(int(highlight_range[0]), int(highlight_range[1]) + 1))
    use_highlight = len(hl_set) > 0
    is_hl = np.isin(parent_id, list(hl_set)) if use_highlight else np.ones(N_METAMER, dtype=bool)
    shuf_size = [15] * 5
    fig, ax = plt.subplots(figsize=(6, 5))
    if use_highlight:
        ax.scatter(x[~is_hl], y[~is_hl], s=10, c='0.82', alpha=0.22, edgecolors='none', zorder=1)
    for shuf in range(5):
        for ani_flag, color in ((True, COLOR_ANI), (False, COLOR_INANI)):
            m = (shuffle == shuf) & (is_ani == ani_flag) & is_hl
            ax.scatter(x[m], y[m], s=shuf_size[shuf], c=color, alpha=SHUF_ALPHA_DEMO[shuf],
                       edgecolors=None, linewidths=0.35, zorder=3 + shuf)
    xl = np.linspace(x.min(), x.max(), 100)
    m_fit, b_fit = np.polyfit(x, y, 1)
    ax.plot(xl, m_fit * xl + b_fit, 'k-', lw=1.5, zorder=2, alpha=0.7)
    ax.set_xlabel('Loading on preferred axis')
    ax.set_ylabel('Response (spikes)')
    hl_note = f'  highlight id={sorted(hl_set)}' if use_highlight else ''
    ax.set_title(f'{area} cell {pc}  R²={r2_demo:.3f}  (N={N_METAMER}){hl_note}')
    ax.grid(True, ls=':', lw=0.5, alpha=0.4)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_TEST_RSP, f'demo_cell{pc}_load_rsp', area=area)

    rsp_all = rsp
    n_cell = meta_load.shape[0]
    axis_norm = np.linalg.norm(axes_fit, axis=1)
    valid_axis = axis_norm > 1e-8
    unit_load = np.full_like(meta_load, np.nan, dtype=np.float32)
    unit_load[valid_axis] = meta_load[valid_axis] / axis_norm[valid_axis, None]
    rsp_plot = rsp_all.astype(np.float32).copy()
    if norm_mode == 'cell_minmax':
        for i in range(n_cell):
            lo, hi = np.nanpercentile(rsp_all[i], cell_norm_pct)
            if hi - lo > 1e-8:
                rsp_plot[i] = np.clip((rsp_all[i] - lo) / (hi - lo), 0, 1)
    hm = np.full((n_cell, n_bins), np.nan, np.float32)
    edges = np.linspace(-1, 1, n_bins + 1)
    for i in range(n_cell):
        if not valid_axis[i]:
            continue
        lo, hi = np.percentile(unit_load[i], [2.5, 97.5])
        sc = np.clip(2 * (unit_load[i] - lo) / (hi - lo + 1e-8) - 1, -1, 1)
        for b in range(n_bins):
            m = (sc >= edges[b]) & (sc < edges[b + 1]) if b < n_bins - 1 else (sc >= edges[b]) & (sc <= edges[b + 1])
            if m.any():
                hm[i, b] = rsp_plot[i, m].mean()
    if norm_mode == 'global_p95':
        global_denom = np.nanpercentile(hm, 95)
        if global_denom > 1e-8:
            hm = np.clip(hm / global_denom, 0, 1)
    elif norm_mode == 'p95':
        for i in range(n_cell):
            if not np.isfinite(hm[i]).any():
                continue
            denom = np.nanpercentile(hm[i], 95)
            if denom > 1e-8:
                hm[i] = np.clip(hm[i] / denom, 0, 1)
    elif norm_mode == 'zscore':
        for i in range(n_cell):
            mu, sd = np.nanmean(hm[i]), np.nanstd(hm[i])
            if sd > 1e-8:
                hm[i] = (hm[i] - mu) / sd
    r2_all = d3['r2']
    if sort_cells == 'r2':
        order = np.argsort(np.nan_to_num(r2_all, nan=-np.inf))[::-1]
    else:
        order = np.arange(n_cell)
    hm = hm[order]
    if show_cells is not None:
        hm = hm[:int(show_cells)]
    if norm_mode in ('cell_minmax', 'p95', 'global_p95'):
        cmap, vmin, vmax = 'Reds', 0, 1
        cbar_label = 'Norm. resp.'
    else:
        cmap, vmin, vmax = 'RdBu_r', -2, 2
        cbar_label = 'Norm. resp. (z per cell)'
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(hm, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='lower', extent=[-1, 1, -0.5, hm.shape[0] - 0.5])
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Distance along preferred axis  ([−1, 1] = 95% stimuli)')
    ax.set_ylabel('Cell (sorted by R²)' if sort_cells == 'r2' else 'Cell')
    ax.set_title(f'{area}  ramp tuning  N={hm.shape[0]}  norm={norm_mode}')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'pop_ramp_heatmap', area=area)


def plot_thought_shared(savepath, meta_paths, n_extreme=5):
    d1 = np.load(shared_path(savepath, 'step1'), allow_pickle=True)
    d2 = np.load(shared_path(savepath, 'step2'), allow_pickle=True)
    nsd_coords = d1['coords']
    meta_coords = d2['coords']
    idx, _, shuffle, is_ani, parent_id, plot_idx = _stim_labels()

    ani_objs = list(range(20))
    inani_objs = list(range(20, 40))
    all_objs = list(range(40))
    w_ani, r2_ani, raw_ani = _fit_shuffle_axis(meta_coords, ani_objs)
    w_inani, r2_inani, raw_inani = _fit_shuffle_axis(meta_coords, inani_objs)
    w_all, r2_all, raw_all = _fit_shuffle_axis(meta_coords, all_objs)
    load_ani = _loadings(meta_coords, w_ani, raw_ani, ani_objs)
    load_inani = _loadings(meta_coords, w_inani, raw_inani, inani_objs)
    load_all = _loadings(meta_coords, w_all, raw_all, all_objs)
    r2_ani_each = _per_obj_r2(load_ani, ani_objs)
    r2_inani_each = _per_obj_r2(load_inani, inani_objs)
    r2_all_each = _per_obj_r2(load_all, all_objs)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, r2_each, r2_g, label, color in zip(
        axes, [r2_ani_each, r2_inani_each], [r2_ani, r2_inani], ['Ani', 'Inani'], [COLOR_ANI, COLOR_INANI],
    ):
        valid = r2_each[np.isfinite(r2_each)]
        ax.hist(valid, bins=12, color=color, alpha=0.55, edgecolor='white', lw=0.6)
        ax.axvline(np.median(valid), color='k', ls='--', lw=1.2, label=f'obj median = {np.median(valid):.3f}')
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
    ax.axvline(np.median(valid), color='k', ls='--', lw=1.2, label=f'obj median = {np.median(valid):.3f}')
    ax.axvline(r2_all, color='C1', ls=':', lw=1.5, label=f'group R² = {r2_all:.3f}')
    ax.set_xlabel('R² (shuffle ~ axis loading)')
    ax.set_ylabel('Object count')
    ax.set_title(f'All  (N_obj = {len(all_objs)})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, 'shuffle_r2_all')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, w, obj_ids, color_base, title in zip(
        axes, [w_ani, w_inani], [ani_objs, inani_objs], [COLOR_ANI, COLOR_INANI],
        ['Ani shuffle axis', 'Inani shuffle axis'],
    ):
        ax.scatter(nsd_coords[:, 0], nsd_coords[:, 1], s=5, c='#dddddd', edgecolors='none', rasterized=True, zorder=1)
        for shuf, alpha in zip(range(5), SHUF_ALPHA):
            m = plot_idx & (shuffle == shuf) & np.isin(parent_id, obj_ids)
            ax.scatter(meta_coords[m, 0], meta_coords[m, 1], s=22, c=color_base, alpha=alpha,
                       edgecolors='white', linewidths=0.25, zorder=3 + shuf, rasterized=True)
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
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, 'shuffle_axis_pc12_ani_inani')

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(nsd_coords[:, 0], nsd_coords[:, 1], s=5, c='#dddddd', edgecolors='none', rasterized=True, zorder=1)
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
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, 'shuffle_axis_pc12_all')

    for tag, w, loads, obj_ids, r2_g in [
        ('ani', w_ani, load_ani, ani_objs, r2_ani),
        ('inani', w_inani, load_inani, inani_objs, r2_inani),
        ('all', w_all, load_all, all_objs, r2_all),
    ]:
        _plot_shuffle_tuning(savepath, meta_paths, loads, obj_ids, shuffle, plot_idx, r2_g, tag, n_extreme)


def _plot_shuffle_tuning(savepath, meta_paths, loads, obj_ids, shuffle, plot_idx, r2_g, tag, n_extreme):
    m_cycle = plot_idx & np.isin(np.arange(N_METAMER) % 40, obj_ids)
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
    parent_id = np.arange(N_METAMER) % 40
    for obj in obj_ids:
        rank = obj_rank[obj]
        base = parent_colors[rank]
        for s in range(5):
            stim = obj + s * 40
            if not plot_idx[stim]:
                continue
            c = _color_by_shuffle(base, s)
            ax.scatter(loads[stim], shuffle[stim], s=28, c=[c], edgecolors='white', linewidths=0.35, zorder=3 + s)
    xl = np.linspace(x.min(), x.max(), 100)
    coef = np.polyfit(x, y, 1)
    ax.plot(xl, np.polyval(coef, xl), 'k-', lw=1.5, alpha=0.75, zorder=2)
    load_s4 = np.array([loads[o + 4 * 40] for o in obj_ids])
    order = np.argsort(load_s4)
    lo_parents = [obj_ids[i] for i in order[:n_extreme]]
    hi_parents = [obj_ids[i] for i in order[-n_extreme:][::-1]]
    for i, p in enumerate(lo_parents):
        stim = int(p)
        edge_c = _color_by_shuffle(parent_colors[obj_rank[p]], 0)
        axes_lo[i].imshow(Image.open(meta_paths[stim]))
        axes_lo[i].set_xticks([])
        axes_lo[i].set_yticks([])
        for sp in axes_lo[i].spines.values():
            sp.set_edgecolor(edge_c)
            sp.set_linewidth(1.2)
    for i, p in enumerate(hi_parents):
        stim = int(p + 4 * 40)
        edge_c = _color_by_shuffle(parent_colors[obj_rank[p]], 4)
        axes_hi[i].imshow(Image.open(meta_paths[stim]))
        axes_hi[i].set_xticks([])
        axes_hi[i].set_yticks([])
        for sp in axes_hi[i].spines.values():
            sp.set_edgecolor(edge_c)
            sp.set_linewidth(1.2)
    ax.set_xlabel('Loading on shuffle axis')
    ax.set_ylabel('Shuffle level')
    ax.set_yticks(range(5))
    ax.set_yticklabels(SHUF_LABELS)
    ax.set_title(f'{tag}  group R²={r2_g:.3f}')
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, f'shuffle_tuning_{tag}')


def plot_thought_area(savepath, cell_rootpath, area, plot_cell=850, n_extreme=5):
    if not os.path.isfile(resolve_area_path(savepath, area, 'shuffle_neuron')):
        print(f'[{area}] skip Thought_Reversed figures — missing shuffle_neuron')
        return

    d2 = np.load(shared_path(savepath, 'step2'), allow_pickle=True)
    meta_paths = list(d2['img_paths'])
    dn = np.load(resolve_area_path(savepath, area, 'shuffle_neuron'), allow_pickle=True)
    rsp = np.load(rsp_path(cell_rootpath, area))
    ds = np.load(shared_path(savepath, 'shuffle_axis'), allow_pickle=True)
    pc = int(plot_cell)
    idx, _, shuffle, is_ani, _, _ = _stim_labels()
    mask_ani, mask_inani = is_ani, ~is_ani
    mask_all = np.ones(N_METAMER, dtype=bool)
    ani_objs = list(range(20))
    inani_objs = list(range(20, 40))
    all_objs = list(range(40))

    r2_a = dn['r2_shuf_ani']
    r2_i = dn['r2_shuf_inani']
    ang_a = dn['angle_ani']
    ang_i = dn['angle_inani']
    r2_all_pop = dn['r2_shuf_all']
    ang_all_pop = dn['angle_all']
    n_cell = rsp.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    for ax, r2, label, color in zip(axes, [r2_a, r2_i], ['Ani shuffle', 'Inani shuffle'], [COLOR_ANI, COLOR_INANI]):
        valid = np.isfinite(r2)
        ax.scatter(np.where(valid)[0], r2[valid], s=8, c=color, alpha=0.55, edgecolors='none', rasterized=True)
        ax.axhline(np.nanmedian(r2), color='k', ls='--', lw=1.0, label=f'median = {np.nanmedian(r2):.3f}')
        ax.set_ylabel('R² (rsp ~ shuffle load)')
        ax.set_title(f'{area}  {label}')
        ax.legend(fontsize=8, loc='upper right')
    axes[1].set_xlabel('Cell index')
    fig.suptitle('Per-neuron shuffle-axis encoding', fontsize=10)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_r2_scatter', area=area)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, ang, label, color in zip(axes, [ang_a, ang_i], ['Ani shuffle axis', 'Inani shuffle axis'], [COLOR_ANI, COLOR_INANI]):
        valid = np.isfinite(ang)
        ax.hist(ang[valid], bins=30, color=color, alpha=0.55, edgecolor='white', lw=0.6)
        ax.axvline(np.nanmedian(ang), color='k', ls='--', lw=1.2, label=f'median = {np.nanmedian(ang):.1f}°')
        ax.set_xlabel('Angle (°)  cell axis vs shuffle axis')
        ax.set_ylabel('Cell count')
        ax.set_title(label)
        ax.legend(fontsize=8)
    fig.suptitle(f'{area}  axis alignment', fontsize=10)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_angle_hist', area=area)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, r2, label, color in zip(axes, [r2_a, r2_i], ['Ani', 'Inani'], [COLOR_ANI, COLOR_INANI]):
        valid = r2[np.isfinite(r2)]
        ax.hist(valid, bins=30, color=color, alpha=0.55, edgecolor='white', lw=0.6)
        ax.axvline(np.median(valid), color='k', ls='--', lw=1.2, label=f'median = {np.median(valid):.3f}')
        ax.set_xlabel('R² (rsp ~ shuffle load)')
        ax.set_ylabel('Cell count')
        ax.set_title(f'{label}  N = {n_cell}')
        ax.legend(fontsize=8)
    fig.suptitle(f'{area}  shuffle-axis R² distribution', fontsize=10)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_r2_hist', area=area)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    for ax, ang, r2, label, color in zip(axes, [ang_a, ang_i], [r2_a, r2_i], ['Ani', 'Inani'], [COLOR_ANI, COLOR_INANI]):
        m = np.isfinite(ang) & np.isfinite(r2)
        ax.scatter(ang[m], r2[m], s=10, c=color, alpha=0.45, edgecolors='none', rasterized=True)
        ax.set_xlabel('Angle (°)')
        ax.set_ylabel('R² (rsp ~ shuffle load)')
        ax.set_title(label)
        ax.grid(True, ls=':', lw=0.5, alpha=0.4)
    fig.suptitle(f'{area}  alignment vs encoding', fontsize=10)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_alignment_vs_encoding', area=area)

    for name, r2_pop, ang_pop, color in [
        ('all_r2_scatter', r2_all_pop, None, COLOR_ALL),
        ('all_angle_hist', None, ang_all_pop, COLOR_ALL),
        ('all_r2_hist', r2_all_pop, None, COLOR_ALL),
    ]:
        if name == 'all_r2_scatter':
            fig, ax = plt.subplots(figsize=(8, 2.5))
            valid = np.isfinite(r2_pop)
            ax.scatter(np.where(valid)[0], r2_pop[valid], s=8, c=color, alpha=0.55, edgecolors='none', rasterized=True)
            ax.axhline(np.nanmedian(r2_pop), color='k', ls='--', lw=1.0, label=f'median = {np.nanmedian(r2_pop):.3f}')
            ax.set_xlabel('Cell index')
            ax.set_ylabel('R² (rsp ~ shuffle load)')
            ax.set_title(f'{area}  All shuffle (40 obj, N=1000)')
            ax.legend(fontsize=8, loc='upper right')
        elif name == 'all_angle_hist':
            fig, ax = plt.subplots(figsize=(4, 3.2))
            valid = np.isfinite(ang_pop)
            ax.hist(ang_pop[valid], bins=30, color=color, alpha=0.55, edgecolor='white', lw=0.6)
            ax.axvline(np.nanmedian(ang_pop), color='k', ls='--', lw=1.2, label=f'median = {np.nanmedian(ang_pop):.1f}°')
            ax.set_xlabel('Angle (°)  cell axis vs shuffle axis')
            ax.set_ylabel('Cell count')
            ax.set_title('All shuffle axis')
            ax.legend(fontsize=8)
        else:
            fig, ax = plt.subplots(figsize=(4, 3.2))
            valid = r2_pop[np.isfinite(r2_pop)]
            ax.hist(valid, bins=30, color=color, alpha=0.55, edgecolor='white', lw=0.6)
            ax.axvline(np.median(valid), color='k', ls='--', lw=1.2, label=f'median = {np.median(valid):.3f}')
            ax.set_xlabel('R² (rsp ~ shuffle load)')
            ax.set_ylabel('Cell count')
            ax.set_title(f'All  N = {n_cell}')
            ax.legend(fontsize=8)
        fig.tight_layout()
        finish_fig(fig, savepath, SCRIPT_THOUGHT, f'pop_shuffle_{name}', area=area)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    m = np.isfinite(ang_all_pop) & np.isfinite(r2_all_pop)
    ax.scatter(ang_all_pop[m], r2_all_pop[m], s=10, c=COLOR_ALL, alpha=0.45, edgecolors='none', rasterized=True)
    ax.set_xlabel('Angle (°)')
    ax.set_ylabel('R² (rsp ~ shuffle load)')
    ax.set_title('All')
    ax.grid(True, ls=':', lw=0.5, alpha=0.4)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, 'pop_shuffle_all_alignment', area=area)

    load_ani, load_inani, load_all = ds['load_ani'], ds['load_inani'], ds['load_all']
    for tag, loads, obj_ids, mask, color in [
        ('ani', load_ani, ani_objs, mask_ani, COLOR_ANI),
        ('inani', load_inani, inani_objs, mask_inani, COLOR_INANI),
        ('all', load_all, all_objs, mask_all, COLOR_ALL),
    ]:
        _demo_shuffle_cell_fig(
            savepath, area, meta_paths, rsp, pc, loads, obj_ids, mask, color, tag,
            float(dn[f'r2_shuf_{tag}'][pc]), float(dn[f'angle_{tag}'][pc]),
            color_by_ani=(tag == 'all'), n_extreme=n_extreme,
        )


def _demo_shuffle_cell_fig(savepath, area, meta_paths, rsp, pc, loads, obj_ids, mask, color,
                           title_tag, r2_val, ang_val, color_by_ani=False, n_extreme=5):
    if color_by_ani:
        x_fit, y_fit = loads, rsp[pc]
    else:
        x_fit, y_fit = loads[mask], rsp[pc, mask]
    xl = np.linspace(x_fit.min(), x_fit.max(), 100)
    coef = np.polyfit(x_fit, y_fit, 1)
    load_s4 = np.array([loads[o + 4 * 40] for o in obj_ids])
    order = np.argsort(load_s4)
    lo_parents = [obj_ids[i] for i in order[:n_extreme]]
    hi_parents = [obj_ids[i] for i in order[-n_extreme:][::-1]]
    lo_stim = [int(p) for p in lo_parents]
    hi_stim = [int(p + 4 * 40) for p in hi_parents]
    fig = plt.figure(figsize=(7.5, 5.5))
    gs = fig.add_gridspec(n_extreme, 3, width_ratios=[1, 3.2, 1], hspace=0.12, wspace=0.08)
    ax = fig.add_subplot(gs[:, 1])
    axes_lo = [fig.add_subplot(gs[i, 0]) for i in range(n_extreme)]
    axes_hi = [fig.add_subplot(gs[i, 2]) for i in range(n_extreme)]
    idx, _, _, is_ani, _, _ = _stim_labels()
    mask_ani, mask_inani = is_ani, ~is_ani
    if color_by_ani:
        ax.scatter(loads[mask_ani], rsp[pc, mask_ani], s=10, alpha=0.35, c=COLOR_ANI, edgecolors='none', zorder=1)
        ax.scatter(loads[mask_inani], rsp[pc, mask_inani], s=10, alpha=0.35, c=COLOR_INANI, edgecolors='none', zorder=1)
    else:
        ax.scatter(x_fit, y_fit, s=10, alpha=0.28, c='0.78', edgecolors='none', zorder=1)
    ax.plot(xl, np.polyval(coef, xl), color=color, lw=1.5, zorder=2)
    ax.set_xlabel('Loading on shuffle axis')
    ax.set_ylabel('Response (spikes)')
    ax.set_title(f'{area} cell {pc}  {title_tag}  R²={r2_val:.3f}  angle={ang_val:.1f}°')
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_THOUGHT, f'demo_cell{pc}_shuffle_{title_tag}', area=area)


def plot_mediation_area(savepath, cell_rootpath, area, demo_ani_obj=17, demo_inani_obj=26,
                        slope_sign=1, rank_ani=2, rank_inani=5, n_obj_show=4, rsp_unit='spikes'):
    med_path = resolve_area_path(savepath, area, 'mediation')
    if not os.path.isfile(med_path):
        print(f'[{area}] skip mediation figures — missing mediation.npz')
        return

    dm = np.load(med_path, allow_pickle=True)
    d3 = np.load(resolve_area_path(savepath, area, 'obj_axis_fit'), allow_pickle=True)
    dn = np.load(resolve_area_path(savepath, area, 'shuffle_neuron'), allow_pickle=True)
    rsp = np.load(rsp_path(cell_rootpath, area))
    meta_load = d3['meta_load']
    r2_load = dm['r2_load']
    avg_load = dm['avg_load']
    avg_rsp = dm['avg_rsp']
    slope_load = dm['slope_load']
    slope_rsp = dm['slope_rsp']
    pearson_r = dm['pearson_r']
    cor_per_obj = dm['cor_per_obj']
    delta_r2 = dm['delta_r2']
    r2_full = dm['r2_full']
    angle_ani = dn['angle_ani']
    angle_inani = dn['angle_inani']
    n_cell = rsp.shape[0]
    _, _, shuffle, is_ani, _, _ = _stim_labels()

    is_ani_obj = np.arange(N_OBJ) < 20
    sl_flat = slope_load.ravel()
    sr_flat = slope_rsp.ravel()
    ani_flat = np.tile(is_ani_obj, n_cell)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.scatter(sl_flat[ani_flat], sr_flat[ani_flat], s=4, alpha=0.15, c=COLOR_ANI, edgecolors='none', rasterized=True, label='Ani')
    ax.scatter(sl_flat[~ani_flat], sr_flat[~ani_flat], s=4, alpha=0.15, c=COLOR_INANI, edgecolors='none', rasterized=True, label='Inani')
    valid = np.isfinite(sl_flat) & np.isfinite(sr_flat)
    lr = sp_stats.linregress(sl_flat[valid], sr_flat[valid])
    xl = np.array([sl_flat[valid].min(), sl_flat[valid].max()])
    ax.plot(xl, lr.slope * xl + lr.intercept, 'k-', lw=1.8, zorder=5, label='fit')
    ax.plot(xl, xl, 'k--', lw=1.0, alpha=0.4, label='slope=1 (expected)')
    ax.set_xlabel('slope_load  (Δload per shuffle unit)')
    ax.set_ylabel(f'slope_rsp   (Δ{rsp_unit} per shuffle unit)')
    ax.set_title(f'{area}  slope scatter  N={valid.sum()}')
    ax.legend(fontsize=8, markerscale=3, loc='upper left')
    ax.grid(True, ls=':', lw=0.5, alpha=0.4)
    ax = axes[1]
    valid_r = pearson_r[np.isfinite(pearson_r)]
    ax.hist(valid_r, bins=40, color='0.45', edgecolor='white', lw=0.5)
    ax.axvline(np.nanmedian(pearson_r), color='C1', ls='--', lw=1.5, label=f'median = {np.nanmedian(pearson_r):.3f}')
    ax.set_xlabel('Pearson r  (slope_load vs slope_rsp,  40 objects)')
    ax.set_ylabel('Cell count')
    ax.set_title(f'{area}  per-neuron r  N={len(valid_r)}')
    ax.legend(fontsize=8)
    fig.suptitle('Test 1: Shuffle slope correlation — load drives rsp', fontsize=10)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_MEDIATION, 'test1_slope_scatter', area=area)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    dr = delta_r2[np.isfinite(delta_r2)]
    axes[0].hist(dr, bins=40, color='0.45', edgecolor='white', lw=0.5)
    axes[0].axvline(np.median(dr), color='C1', ls='--', lw=1.5, label=f'median={np.median(dr):.4f}')
    axes[0].axvline(0, color='k', ls=':', lw=1.0)
    axes[0].set_xlabel('ΔR²  (R²_full − R²_load)')
    axes[0].set_ylabel('Cell count')
    axes[0].set_title('A: ΔR² distribution')
    axes[0].legend(fontsize=8)
    m = np.isfinite(delta_r2) & np.isfinite(r2_load)
    axes[1].scatter(r2_load[m], delta_r2[m], s=8, alpha=0.4, c='0.4', edgecolors='none', rasterized=True)
    axes[1].axhline(0, color='C1', ls='--', lw=1.0)
    axes[1].set_xlabel('R²_load  (object space model)')
    axes[1].set_ylabel('ΔR²  (additional from shuffle)')
    axes[1].set_title('B: R²_load vs ΔR²')
    axes[1].grid(True, ls=':', lw=0.5, alpha=0.4)
    fig.suptitle(f'{area}  Test 2: Incremental ΔR²', fontsize=10)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_MEDIATION, 'test2_delta_r2', area=area)

    demo_cell = int(np.nanargmax(r2_load))
    pc = demo_cell
    x = meta_load[pc]
    y = rsp[pc]
    m_fit, b_fit = np.polyfit(x, y, 1)
    resid = y - (m_fit * x + b_fit)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=8, alpha=0.28, c='0.75', edgecolors='none', zorder=1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, m_fit * xl + b_fit, 'r-', lw=2, zorder=3)
    ax.set_xlabel('Loading on preferred axis')
    ax.set_ylabel(f'Response ({rsp_unit})')
    ax.set_title(f'{area} cell {pc}  A: Load → Rsp  R²={r2_load[pc]:.3f}')
    ax.grid(True, ls=':', lw=0.5, alpha=0.4)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_MEDIATION, f'demo_cell{pc}_panel_A', area=area)

    half = n_obj_show // 2
    ani_objs = np.arange(20)
    inani_objs = np.arange(20, 40)
    sel_ani = ani_objs[np.argsort(np.abs(slope_rsp[pc, ani_objs]))[-half:]]
    sel_inani = inani_objs[np.argsort(np.abs(slope_rsp[pc, inani_objs]))[-half:]]
    shuf_x = np.arange(N_SHUF)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
    for k, o in enumerate(sel_ani):
        c = plt.cm.Reds(0.35 + 0.55 * k / (half - 1 + 1e-8))
        ax_top.plot(shuf_x, avg_load[pc, o, :], '-o', color=c, ms=4, lw=1.4, alpha=0.85)
        ax_bot.plot(shuf_x, avg_rsp[pc, o, :], '-o', color=c, ms=4, lw=1.4, alpha=0.85)
    for k, o in enumerate(sel_inani):
        c = plt.cm.Blues(0.35 + 0.55 * k / (half - 1 + 1e-8))
        ax_top.plot(shuf_x, avg_load[pc, o, :], '--s', color=c, ms=4, lw=1.4, alpha=0.85)
        ax_bot.plot(shuf_x, avg_rsp[pc, o, :], '--s', color=c, ms=4, lw=1.4, alpha=0.85)
    ax_top.set_ylabel('Avg load on prefer axis')
    ax_top.set_title(f'{area} cell {pc}  B: Per-object trajectories across shuffle')
    ax_bot.set_ylabel('Avg firing rate (spikes)')
    ax_bot.set_xlabel('Shuffle level')
    ax_bot.set_xticks(shuf_x)
    ax_bot.set_xticklabels(SHUF_LABELS)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_MEDIATION, f'demo_cell{pc}_panel_B', area=area)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.scatter(shuffle + np.random.randn(N_METAMER) * 0.07, resid, s=6, alpha=0.18, c='0.55', edgecolors='none', rasterized=True)
    for flag, color, label in ((True, COLOR_ANI, 'Ani'), (False, COLOR_INANI, 'Inani')):
        m_r, b_r = np.polyfit(shuffle[is_ani == flag], resid[is_ani == flag], 1)
        xl2 = np.array([0, 4], dtype=float)
        ax.plot(xl2, m_r * xl2 + b_r, '-', color=color, lw=2, label=f'{label} slope={m_r:.3f}')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Shuffle level')
    ax.set_ylabel('Residual  (rsp − load model)')
    ax.set_xticks(np.arange(N_SHUF))
    ax.set_xticklabels(SHUF_LABELS)
    ax.set_title(f'{area} cell {pc}  C: Residual vs Shuffle')
    ax.legend(fontsize=8)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_MEDIATION, f'demo_cell{pc}_panel_C', area=area)

    cor_ani_flat = cor_per_obj[:, :20].ravel()
    cor_inani_flat = cor_per_obj[:, 20:].ravel()
    cor_ani_flat = cor_ani_flat[np.isfinite(cor_ani_flat)]
    cor_inani_flat = cor_inani_flat[np.isfinite(cor_inani_flat)]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bins = np.linspace(-1, 1, 35)
    ax.hist(cor_ani_flat, bins=bins, color=COLOR_ANI, alpha=0.55, edgecolor='white', lw=0.5,
            label=f'Ani  (median={np.median(cor_ani_flat):.3f})')
    ax.hist(cor_inani_flat, bins=bins, color=COLOR_INANI, alpha=0.55, edgecolor='white', lw=0.5,
            label=f'Inani (median={np.median(cor_inani_flat):.3f})')
    ax.set_xlabel('Pearson r  (load ~ rsp,  5 shuffle points,  per neuron × object)')
    ax.set_ylabel('Count  (neuron × object pairs)')
    ax.set_title(f'{area}  Load–Rsp coupling: Ani vs Inani objects')
    ax.legend(fontsize=9)
    fig.tight_layout()
    finish_fig(fig, savepath, SCRIPT_MEDIATION, 'test6a_cor_coupling', area=area)

    def _rank_for_obj(obj_idx, sign):
        sc = cor_per_obj[:, obj_idx] * r2_load
        sc = sc.copy().astype(float)
        sc[~np.isfinite(sc)] = np.nan
        if sign != 0:
            sc[slope_load[:, obj_idx] * sign <= 0] = np.nan
        return np.argsort(np.nan_to_num(sc, nan=-np.inf))[::-1]

    ranked_ani = _rank_for_obj(demo_ani_obj, slope_sign)
    ranked_inani = _rank_for_obj(demo_inani_obj, slope_sign)
    pc_ani = int(ranked_ani[rank_ani - 1])
    pc_inani = int(ranked_inani[rank_inani - 1])
    shuf_x = np.arange(N_SHUF)

    for pc_i, obj_idx, color, tag in [
        (pc_ani, demo_ani_obj, COLOR_ANI, 'ani'),
        (pc_inani, demo_inani_obj, COLOR_INANI, 'inani'),
    ]:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8, 3.8))
        for ax, data, ylabel in zip([ax_l, ax_r], [avg_load, avg_rsp],
                                    ['Load on preferred axis', f'Firing rate ({rsp_unit})']):
            ax.plot(shuf_x, data[pc_i, obj_idx, :], '-o', color=color, lw=2.2, ms=8,
                    label=f'{tag} img {obj_idx + 1}')
            ax.set_xticks(shuf_x)
            ax.set_xticklabels(SHUF_LABELS)
            ax.set_xlabel('Shuffle level')
            ax.set_ylabel(ylabel)
            ax.grid(True, ls=':', lw=0.5, alpha=0.4)
            ax.legend(fontsize=9)
        fig.suptitle(f'{area} cell {pc_i}  img {obj_idx + 1}  R²={r2_load[pc_i]:.3f}', fontsize=10)
        fig.tight_layout()
        finish_fig(fig, savepath, SCRIPT_MEDIATION, f'demo_{tag}_cell{pc_i}_obj{obj_idx + 1}', area=area)


def generate_all_figures(savepath, cell_rootpath, areas=None, show=False):
    configure(save=True, show=show)
    areas = areas or BRAIN_AREAS
    print('[figures] shared Test_Obj_Space_Rsp …')
    plot_test_rsp_shared(savepath)
    d2 = np.load(shared_path(savepath, 'step2'), allow_pickle=True)
    meta_paths = list(d2['img_paths'])
    print('[figures] shared Obj_Space_Thought_Reversed …')
    plot_thought_shared(savepath, meta_paths)
    for area in areas:
        if not os.path.isfile(rsp_path(cell_rootpath, area)):
            print(f'[{area}] skip figures — missing rsp')
            continue
        print(f'[{area}] figures …')
        plot_test_rsp_area(savepath, cell_rootpath, area)
        plot_thought_area(savepath, cell_rootpath, area)
        plot_mediation_area(savepath, cell_rootpath, area)
    print('[figures] done.')
