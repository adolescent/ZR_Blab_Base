'''
为每个神经元设定同一个衰减系数，公式为：
Rsp = Rsp0 * (1 - beta * shuffle_level * Rsp0)
拟合中只有一个参数，其它全都是测量量。我们观察到衰减和神经元发放率之间存在联系，并试图描述这个联系

主要是比较M和A侧的脑区是否存在更本质的不同。
'''

#%% Directory and imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import OS_Tools as ot

datapath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\decay_index_unify'

brain_areas = ['ML', 'MSB', 'AL', 'ASB']
WINDOW_S = 0.17
N_REPEAT = 5
N_SHUF = 5
N_IMG = 40
RAW_MIN_HZ = 5
MIN_VALID_IMG = 1
ANI_IMG = np.arange(20)
INANI_IMG = np.arange(20, 40)
FIT_SCOPES = [
    ('all', 'All images', None),
    ('ani', 'Ani images', True),
    ('inani', 'Inani images', False),
]

# shuffle level for each of 25 points: repeat0 shuf0-4, repeat1 shuf0-4, ...
X_SHUF = np.tile(np.arange(N_SHUF, dtype=float), N_REPEAT)
EPS = 1e-12


def p_to_star(p):
    if not np.isfinite(p):
        return 'ns'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'


def plot_violin_box(ax, data, positions, colors, violin_w=0.65, box_w=0.18, whis=(10, 90)):
    """Literature-style transparent violin with narrow box overlay."""
    parts = ax.violinplot(
        data, positions=positions, widths=violin_w,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor(colors[i])
        body.set_alpha(0.35)
    ax.boxplot(
        data, positions=positions, widths=box_w,
        patch_artist=True, showfliers=False, whis=whis,
        medianprops=dict(color='k', lw=1.5),
        boxprops=dict(facecolor='white', edgecolor='k', linewidth=1.2),
        whiskerprops=dict(color='k', linewidth=1),
        capprops=dict(color='k', linewidth=1),
    )


def add_sig_bracket(ax, x1, x2, y, h, text, fs=9):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c='k')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=fs)


#%% Fitting and utility functions

def load_area_response(area):
    """Load one area's responses and return raw rsp plus 25 shuffle-level samples."""
    area_dir = ot.Join(datapath, area)
    rsp = np.load(ot.Join(area_dir, 'avr_rsp.npy'))
    info = pd.read_csv(ot.Join(area_dir, 'cell_site_info.csv'))

    n_cell = rsp.shape[0]
    rsp_hz = rsp / WINDOW_S
    r4 = rsp_hz.reshape(n_cell, N_REPEAT, N_SHUF, N_IMG)

    # Rsp_Raw is the response to level-0 images, averaged across repeats.
    raw = r4[:, :, 0, :].mean(1)
    y = r4.transpose(0, 3, 1, 2).reshape(n_cell, N_IMG, N_REPEAT * N_SHUF)
    return raw, y, rsp_hz, info, r4


def scope_img_ids(scope_flag):
    """Return image indices for all, ani, or inani scope."""
    if scope_flag is None:
        return np.arange(N_IMG)
    if scope_flag:
        return ANI_IMG
    return INANI_IMG


def fit_decay_beta_unified(raw, y, img_ids=None, min_valid_img=MIN_VALID_IMG):
    """
    Fit Rsp = Rsp0 * (1 - beta * shuffle_level * Rsp0) with one beta per cell.

    OLS: target = Rsp0 - Rsp = beta * Rsp0^2 * shuffle_level
    img_ids limits which images enter the fit (all / ani / inani).
    """
    if img_ids is None:
        img_ids = np.arange(N_IMG)
    raw_s = raw[:, img_ids]
    y_s = y[:, img_ids, :]

    x = X_SHUF
    raw_valid = np.isfinite(raw_s) & (raw_s > RAW_MIN_HZ)
    valid = raw_valid[..., None] & np.isfinite(y_s)

    predictor = (raw_s[..., None] ** 2) * x
    target = raw_s[..., None] - y_s

    num = np.where(valid, predictor * target, 0.0).sum(axis=(1, 2))
    den = np.where(valid, predictor ** 2, 0.0).sum(axis=(1, 2))
    n_valid_img = raw_valid.sum(axis=1).astype(int)
    ok_fit = (den > EPS) & (n_valid_img >= min_valid_img)

    n_cell = raw.shape[0]
    beta = np.full(n_cell, np.nan, dtype=float)
    beta[ok_fit] = num[ok_fit] / den[ok_fit]

    y_hat = raw_s[..., None] * (1.0 - beta[:, None, None] * x * raw_s[..., None])
    n_pt = valid.sum(axis=(1, 2)).astype(int)

    sy = np.where(valid, y_s, 0.0).sum(axis=(1, 2))
    y_mean = np.divide(sy, n_pt, out=np.full(n_cell, np.nan, dtype=float), where=n_pt > 0)
    ss_res = np.where(valid, (y_s - y_hat) ** 2, 0.0).sum(axis=(1, 2))
    ss_tot = np.where(valid, (y_s - y_mean[:, None, None]) ** 2, 0.0).sum(axis=(1, 2))
    r2 = 1.0 - ss_res / np.maximum(ss_tot, EPS)
    r2[~ok_fit] = np.nan

    return {
        'beta': beta,
        'r2': r2,
        'n_valid_img': n_valid_img,
        'n_pt': n_pt,
        'raw_valid': raw_valid,
        'img_ids': img_ids,
    }


def build_cell_frame(area, raw, fit, info, scope_name, scope_label):
    """Build one row per neuron with unified beta and fit quality for one image scope."""
    img_ids = fit['img_ids']
    raw_s = raw[:, img_ids]
    n_cell = raw.shape[0]
    raw_valid = fit['raw_valid']
    raw_mean = np.full(n_cell, np.nan, dtype=float)
    has_valid = raw_valid.any(axis=1)
    raw_mean[has_valid] = (
        np.where(raw_valid, raw_s, np.nan).sum(axis=1)[has_valid]
        / raw_valid.sum(axis=1)[has_valid]
    )

    df = pd.DataFrame({
        'area': area,
        'cell_idx': np.arange(n_cell),
        'image_scope': scope_name,
        'image_scope_label': scope_label,
        'beta': fit['beta'],
        'r2': fit['r2'],
        'n_valid_img': fit['n_valid_img'],
        'n_pt': fit['n_pt'],
        'raw_mean': raw_mean,
    })

    meta_cols = ['global_idx', 'site_name', 'dprime_face', 'dprime_body']
    present_meta = [c for c in meta_cols if c in info.columns]
    if present_meta:
        df = df.merge(
            info[present_meta],
            left_on='cell_idx',
            right_index=True,
            how='left',
        )
    return df


def _as_img_list(img_id):
    return [int(img_id)] if np.isscalar(img_id) else [int(i) for i in img_id]


def save_fig(fig, fname, fig_dir):
    ot.Mkdir(fig_dir)
    out = ot.Join(fig_dir, fname)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'saved: {out}')


def _demo_ctx(area, cell_idx, img_id, df_cell, cache, scope_name, fit_scopes,
              color_ani, color_inani):
    """Shared data for the three demo panels."""
    img_ids = _as_img_list(img_id)
    row = df_cell[
        (df_cell['area'] == area) & (df_cell['cell_idx'] == cell_idx)
        & (df_cell['image_scope'] == scope_name)
    ]
    if len(row) == 0:
        return None
    scope_flag = {s[0]: s[2] for s in fit_scopes}[scope_name]
    scope_img = scope_img_ids(scope_flag)
    raw_valid = np.zeros(N_IMG, dtype=bool)
    raw_valid[scope_img] = cache[area]['fits'][scope_name]['raw_valid'][cell_idx]
    return {
        'area': area, 'cell_idx': cell_idx, 'scope_name': scope_name,
        'scope_label': row['image_scope_label'].iloc[0],
        'beta': float(row['beta'].iloc[0]), 'r2': float(row['r2'].iloc[0]),
        'img_ids': img_ids, 'levels': np.arange(N_SHUF, dtype=float),
        'raw_all': cache[area]['raw'][cell_idx],
        'r4': cache[area]['r4'], 'raw_valid': raw_valid,
        'curve_colors': plt.cm.tab10(np.linspace(0, 1, max(len(img_ids), 1))),
        'img_colors': [color_ani if i < 20 else color_inani for i in range(N_IMG)],
        'img_ticks': np.arange(1, N_IMG + 1),
    }


def plot_demo_decay_curves(ax, ctx, ylim=None):
    for i, iid in enumerate(ctx['img_ids']):
        c = ctx['curve_colors'][i]
        raw_img = ctx['r4'][ctx['cell_idx'], :, 0, iid].mean()
        change = raw_img - ctx['r4'][ctx['cell_idx'], :, :, iid].mean(0)
        ax.scatter(ctx['levels'], change, s=40, color=c, zorder=3)
        ax.plot(ctx['levels'], ctx['beta'] * raw_img ** 2 * ctx['levels'],
                '-', color=c, lw=2, label=f'img {iid + 1}: raw={raw_img:.1f} Hz')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xticks(ctx['levels'])
    ax.set_xlabel('Shuffle level')
    ax.set_ylabel('Firing rate change (Hz)')
    img_nos = [i + 1 for i in ctx['img_ids']]
    ax.set_title(f'{ctx["area"]} cell {ctx["cell_idx"]}  {ctx["scope_label"]}  img {img_nos}\n'
                 f'beta={ctx["beta"]:.4f}, R2={ctx["r2"]:.2f}')
    ax.legend(fontsize=7)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_demo_raw_bar(ax, ctx, ylim=None):
    bar_colors = [ctx['img_colors'][i] if ctx['raw_valid'][i] else '0.85' for i in range(N_IMG)]
    ax.bar(ctx['img_ticks'], ctx['raw_all'], color=bar_colors, width=0.8, edgecolor='none')
    ax.axhline(RAW_MIN_HZ, color='k', ls=':', lw=1.0, label=f'>{RAW_MIN_HZ} Hz')
    for i, iid in enumerate(ctx['img_ids']):
        ax.axvline(iid + 1, color=ctx['curve_colors'][i], ls='--', lw=1.2)
    ax.set_xlim(0.5, N_IMG + 0.5)
    ax.set_xlabel('Image')
    ax.set_ylabel('Raw firing rate (Hz)')
    ax.set_title('Raw response  (gray = excluded)')
    ax.legend(fontsize=7)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_demo_decay_bar(ax, ctx, ylim=None, lvl=4):
    emp = ctx['raw_all'] - ctx['r4'][ctx['cell_idx'], :, lvl, :].mean(0)
    pred = ctx['beta'] * ctx['raw_all'] ** 2 * lvl
    ax.bar(ctx['img_ticks'], emp, color=ctx['img_colors'], width=0.8, alpha=0.45,
           edgecolor='none', label='empirical')
    ax.plot(ctx['img_ticks'], pred, 'k.-', ms=4, lw=1.2, label='unified pred')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    for i, iid in enumerate(ctx['img_ids']):
        ax.axvline(iid + 1, color=ctx['curve_colors'][i], ls='--', lw=1.2)
    ax.set_xlim(0.5, N_IMG + 0.5)
    ax.set_xlabel('Image')
    ax.set_ylabel(f'Decay at shuffle level {lvl} (Hz)')
    ax.set_title('Per-image decay vs unified prediction')
    ax.legend(fontsize=7)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_neuron_demo(area, cell_idx, img_id, df_cell, cache, scope_name, fit_scopes,
                     color_ani, color_inani, fig_dir,
                     ylim_decay=None, ylim_raw=None, ylim_decay_bar=None):
    ctx = _demo_ctx(area, cell_idx, img_id, df_cell, cache, scope_name, fit_scopes,
                    color_ani, color_inani)
    if ctx is None:
        print(f'demo skipped: {area} cell {cell_idx} scope={scope_name} not found')
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_demo_decay_curves(axes[0], ctx, ylim=ylim_decay)
    plot_demo_raw_bar(axes[1], ctx, ylim=ylim_raw)
    plot_demo_decay_bar(axes[2], ctx, ylim=ylim_decay_bar)
    fig.suptitle(f'{area}  cell {cell_idx}  scope={scope_name}  |  red=ani  blue=inani', fontsize=11)
    fig.tight_layout()
    tag = '_'.join(str(i + 1) for i in ctx['img_ids'])
    save_fig(fig, f'demo_{area}_cell{cell_idx}_{scope_name}_img{tag}.png', fig_dir)


def plot_metric_by_area_scopes(df, metric, ylabel, title, fname, fig_dir,
                               area_order, compare_pairs, image_scopes, box_whis,
                               ylim=None, bracket_h=0.02, bracket_row_gap=0.02,
                               bracket_top_pad=0.01):
    """Violin/box per area, with all / ani / inani as three hues."""
    area_pos = {a: i for i, a in enumerate(area_order)}
    scope_offsets = np.linspace(-0.27, 0.27, len(image_scopes))
    data, positions, colors = [], [], []
    for i, area in enumerate(area_order):
        for j, (scope_name, _, _, color) in enumerate(image_scopes):
            vals = df.loc[
                (df['area'] == area) & (df['image_scope'] == scope_name) & np.isfinite(df[metric]),
                metric,
            ].to_numpy()
            data.append(vals)
            positions.append(i + scope_offsets[j])
            colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_violin_box(ax, data, positions, colors, violin_w=0.22, box_w=0.06, whis=box_whis)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xticks(range(len(area_order)))
    ax.set_xticklabels(area_order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    y_top = ax.get_ylim()[1]
    for j, (scope_name, scope_label, _, _) in enumerate(image_scopes):
        y_br = y_top - bracket_top_pad - bracket_h - j * (bracket_h + bracket_row_gap)
        off = scope_offsets[j]
        for a1, a2 in compare_pairs:
            v1 = df.loc[(df['area'] == a1) & (df['image_scope'] == scope_name), metric].dropna().to_numpy()
            v2 = df.loc[(df['area'] == a2) & (df['image_scope'] == scope_name), metric].dropna().to_numpy()
            if len(v1) < 3 or len(v2) < 3:
                continue
            _, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            print(f'{fname} [{scope_label}] {a1} vs {a2}: p={p:.4g}, n=({len(v1)}, {len(v2)})')
            add_sig_bracket(ax, area_pos[a1] + off, area_pos[a2] + off, y_br, bracket_h, p_to_star(p))
    ax.legend(handles=[Patch(facecolor=s[3], alpha=0.35, label=s[1]) for s in image_scopes],
              loc='upper right', frameon=False)
    fig.tight_layout()
    save_fig(fig, fname, fig_dir)


def plot_beta_r2_hist(sub, area, scope_name, scope_label, area_colors, fig_dir,
                      ylim_beta=None, ylim_r2=None):
    color = area_colors[area]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label, ylim, bins, xr in [
        (axes[0], 'beta', 'beta', ylim_beta, 60, None),
        (axes[1], 'r2', 'R2', ylim_r2, 50, (0, 1)),
    ]:
        vals = sub.loc[np.isfinite(sub[col]), col].to_numpy()
        ax.hist(vals, bins=bins, range=xr, color=color, alpha=0.75, edgecolor='white')
        if len(vals):
            ax.axvline(np.nanmedian(vals), color='k', ls='--', lw=1.2,
                       label=f'median={np.nanmedian(vals):.4f}' if col == 'beta' else f'median={np.nanmedian(vals):.2f}')
        if col == 'beta':
            ax.axvline(0, color='gray', ls=':', lw=1.0)
        ax.set_xlabel(f'{label}  (unified fit)' if col == 'r2' else 'Beta')
        ax.set_ylabel('Count')
        ax.set_title(f'{area}  {scope_label}  {label}  n={len(vals)}')
        ax.legend(fontsize=8)
        if ylim is not None:
            ax.set_ylim(ylim)
    fig.tight_layout()
    save_fig(fig, f'beta_r2_hist_{area}_{scope_name}.png', fig_dir)


def plot_beta_vs_raw(scope_df, scope_label, scope_name, area_order, area_colors, fig_dir,
                     ylim=None, xlim=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    for area in area_order:
        sub = scope_df[(scope_df['area'] == area) & np.isfinite(scope_df['beta']) & np.isfinite(scope_df['raw_mean'])]
        ax.scatter(sub['raw_mean'], sub['beta'], s=12, color=area_colors[area],
                   alpha=0.35, edgecolors='none', rasterized=True, label=area)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Mean raw firing rate (Hz)')
    ax.set_ylabel('Beta')
    ax.set_title(f'Beta vs mean raw  ({scope_label})')
    ax.legend(fontsize=9, frameon=False)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()
    save_fig(fig, f'beta_vs_raw_mean_{scope_name}.png', fig_dir)


#%% Load, fit unified beta, save per-neuron tables

ot.Mkdir(savepath)

cache = {}
all_dfs = []
for area in brain_areas:
    raw, y, rsp_hz, info, r4 = load_area_response(area)
    fits_by_scope = {}
    area_dfs = []
    out_dir = ot.Join(savepath, area)
    ot.Mkdir(out_dir)

    for scope_name, scope_label, scope_flag in FIT_SCOPES:
        img_ids = scope_img_ids(scope_flag)
        fit = fit_decay_beta_unified(raw, y, img_ids=img_ids)
        fits_by_scope[scope_name] = fit

        df = build_cell_frame(area, raw, fit, info, scope_name, scope_label)
        df.to_csv(ot.Join(out_dir, f'decay_beta_unified_{scope_name}.csv'), index=False)
        area_dfs.append(df)
        n_fit = np.isfinite(fit['beta']).sum()
        print(f'{area} {scope_name}: {raw.shape[0]} cells, {n_fit} fitted')

    cache[area] = {
        'raw': raw, 'y': y, 'rsp_hz': rsp_hz, 'r4': r4,
        'fits': fits_by_scope, 'fit': fits_by_scope['all'],
    }
    all_dfs.extend(area_dfs)

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv(ot.Join(savepath, 'decay_beta_unified_all.csv'), index=False)
print(f'saved all: {len(df_all)} rows -> {ot.Join(savepath, "decay_beta_unified_all.csv")}')

for scope_name, _, _ in FIT_SCOPES:
    scope_df = df_all[df_all['image_scope'] == scope_name]
    scope_df.to_csv(ot.Join(savepath, f'decay_beta_unified_{scope_name}.csv'), index=False)
    print(f'saved {scope_name}: {len(scope_df)} rows')


#%% Plot parameters & figures

FIG_DIR = ot.Join(savepath, 'figures', 'Decay_Index_Unify')
ot.Mkdir(FIG_DIR)

# --- layout ---
AREA_ORDER = ['AL', 'ML', 'ASB', 'MSB']          # x-axis order
AREA_COLORS = dict(zip(AREA_ORDER, ['#27ae60', '#c0392b', '#8e44ad', '#2980b9']))
COMPARE_PAIRS = [('ASB', 'MSB'), ('AL', 'ML')]   # significance brackets
plot_areas = AREA_ORDER                          # areas for per-area hist plots
BOX_WHIS = (10, 90)

# --- image scope hues ---
COLOR_ALL = '#2c3e50'
COLOR_ANI = '#c0392b'
COLOR_INANI = '#2980b9'
SCOPE_COLORS = {'all': COLOR_ALL, 'ani': COLOR_ANI, 'inani': COLOR_INANI}
IMAGE_SCOPES = [
    (name, label, flag, SCOPE_COLORS[name])
    for name, label, flag in FIT_SCOPES
]

# --- axis limits ---
YLIM = {
    'beta_by_area': (-0.005, 0.06 ),
    'r2_by_area': (0, 1),
    'hist_beta': None,
    'hist_r2': None,
    'beta_vs_raw': None,
    'beta_vs_raw_x': None,
    'demo_decay': None,
    'demo_raw': None,
    'demo_decay_bar': None,
}
# significance brackets: bracket_h=bar height, bracket_row_gap=gap between scope rows, bracket_top_pad=margin below y-top
BETA_BRACKETS = dict(bracket_h=0.003, bracket_row_gap=0.005, bracket_top_pad=0.005)
R2_BRACKETS = dict(bracket_h=0.03, bracket_row_gap=0.02, bracket_top_pad=0.01)

# --- demo neuron ---
DEMO_AREA = 'ASB'
DEMO_CELL = 902
DEMO_IMG = [8, 12, 31]
DEMO_SCOPE = 'all'

# --- run plots ---
plot_metric_by_area_scopes(
    df_all, 'beta', 'Beta', 'Beta by area (all / ani / inani)', 'beta_by_area.png',
    FIG_DIR, AREA_ORDER, COMPARE_PAIRS, IMAGE_SCOPES, BOX_WHIS,
    YLIM['beta_by_area'], **BETA_BRACKETS,
)
plot_metric_by_area_scopes(
    df_all, 'r2', 'R2', 'R2 by area (all / ani / inani)', 'r2_by_area.png',
    FIG_DIR, AREA_ORDER, COMPARE_PAIRS, IMAGE_SCOPES, BOX_WHIS,
    YLIM['r2_by_area'], **R2_BRACKETS,
)

for scope_name, scope_label, _, _ in IMAGE_SCOPES:
    sdf = df_all[df_all['image_scope'] == scope_name]
    for area in plot_areas:
        plot_beta_r2_hist(
            sdf[sdf['area'] == area], area, scope_name, scope_label,
            AREA_COLORS, FIG_DIR, YLIM['hist_beta'], YLIM['hist_r2'],
        )
    plot_beta_vs_raw(
        sdf, scope_label, scope_name, AREA_ORDER, AREA_COLORS, FIG_DIR,
        YLIM['beta_vs_raw'], YLIM['beta_vs_raw_x'],
    )

plot_neuron_demo(
    DEMO_AREA, DEMO_CELL, DEMO_IMG, df_all, cache, DEMO_SCOPE, FIT_SCOPES,
    COLOR_ANI, COLOR_INANI, FIG_DIR,
    ylim_decay=YLIM['demo_decay'], ylim_raw=YLIM['demo_raw'],
    ylim_decay_bar=YLIM['demo_decay_bar'],
)

#%%
