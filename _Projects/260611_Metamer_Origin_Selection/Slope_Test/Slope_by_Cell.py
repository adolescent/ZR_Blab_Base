
#%% 目录和 import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import OS_Tools as ot

datapath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\slope'

brain_areas = ['ML', 'MSB', 'AL', 'ASB']
plot_areas = ['ML', 'MSB', 'AL', 'ASB']
WINDOW_S = 0.17
N_REPEAT = 5
N_SHUF = 5
N_IMG = 40
N_PERM = 200
N_SHIFT = 200
CONTROL_SEED = 0
KDE_MAX_N = 10000          # KDE 绘图子采样上限（回归仍用全量点）
KDE_SEED = 0
FIG_DIR = ot.Join(savepath, 'figures', 'Slope_by_Cell')
COLOR_ANI = '#c0392b'
COLOR_INANI = '#2980b9'
ANI_IMG = np.arange(20)
INANI_IMG = np.arange(20, 40)

# shuffle level for each of 25 points: repeat0 shuf0-4, repeat1 shuf0-4, ...
X_SHUF = np.tile(np.arange(N_SHUF, dtype=float), N_REPEAT)
X2 = float((X_SHUF ** 2).sum())


def fit_from_y(raw, y):
    """raw: (n_cell, 40), y: (n_cell, 40, 25) -> slope, r2."""
    dy = y - raw[:, :, None]
    slope = (dy * X_SHUF).sum(-1) / X2
    ss_res = ((dy - slope[..., None] * X_SHUF) ** 2).sum(-1)
    ss_tot = (dy ** 2).sum(-1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return slope, r2


def fit_raw_anchored_slope(rsp_hz=None, y=None, raw=None):
    """rsp_hz (n_cell,1000) or prebuilt y/raw -> raw, slope, r2, y."""
    if rsp_hz is not None:
        n_cell = rsp_hz.shape[0]
        r4 = rsp_hz.reshape(n_cell, N_REPEAT, N_SHUF, N_IMG)
        raw = r4[:, 0].mean(1)
        y = r4.transpose(0, 3, 1, 2).reshape(n_cell, N_IMG, 25)
    slope, r2 = fit_from_y(raw, y)
    return raw, slope, r2, y


def group_metrics(raw, slope, r2, img_ids):
    """Summary stats for one image group (ani or inani)."""
    sl = slope[:, img_ids].ravel()
    r2v = r2[:, img_ids].ravel()
    rawv = raw[:, img_ids].ravel()
    valid = np.isfinite(rawv) & np.isfinite(sl)
    valid_r2 = np.isfinite(rawv) & np.isfinite(r2v)
    corr_sl = np.corrcoef(rawv[valid], sl[valid])[0, 1] if valid.sum() > 2 else np.nan
    corr_r2 = np.corrcoef(rawv[valid_r2], r2v[valid_r2])[0, 1] if valid_r2.sum() > 2 else np.nan
    return {
        'median_abs_slope': float(np.nanmedian(np.abs(sl))),
        'median_r2': float(np.nanmedian(r2v)),
        'corr_raw_slope': float(corr_sl),
        'corr_raw_r2': float(corr_r2),
    }


def null_p_value(null_vals, obs_val, use_abs=False):
    """One-sided p: fraction of null >= observed (or |null| >= |obs|)."""
    null_vals = np.asarray(null_vals, dtype=float)
    if use_abs:
        return float(np.mean(np.abs(null_vals) >= abs(obs_val)))
    return float(np.mean(null_vals >= obs_val))


def plot_scatter_regress(ax, sub, xcol, ycol, is_ani, color, label):
    """Scatter + linregress line for one ani/inani group."""
    m = sub['is_ani'] == is_ani
    x = sub.loc[m, xcol].to_numpy()
    y = sub.loc[m, ycol].to_numpy()
    ax.scatter(x, y, s=6, color=color, alpha=0.25, edgecolors='none', rasterized=True)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() > 2:
        lr = stats.linregress(x[valid], y[valid])
        xl = np.array([np.nanmin(x[valid]), np.nanmax(x[valid])])
        ax.plot(xl, lr.slope * xl + lr.intercept, '-', color=color, lw=2)
        return (f'{label}: r={lr.rvalue:.3f}, p={lr.pvalue:.2e}, '
                f'slope={lr.slope:.4f}')
    return f'{label}: n<3'


def plot_kde_regress(ax, sub, xcol, ycol, is_ani, color, label, rng=None):
    """2D KDE (subsampled) + linregress on all points for one ani/inani group."""
    m = sub['is_ani'] == is_ani
    x = sub.loc[m, xcol].to_numpy()
    y = sub.loc[m, ycol].to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) > KDE_MAX_N:
        rng = np.random.default_rng(KDE_SEED) if rng is None else rng
        idx = rng.choice(len(x), KDE_MAX_N, replace=False)
        x_kde, y_kde = x[idx], y[idx]
    else:
        x_kde, y_kde = x, y
    sns.kdeplot(x=x_kde, y=y_kde, ax=ax, color=color, fill=True, alpha=0.2,
                levels=6, thresh=0.08, label=label)
    sns.kdeplot(x=x_kde, y=y_kde, ax=ax, color=color, fill=False,
                levels=6, thresh=0.08, linewidths=1.2)
    if len(x) > 2:
        lr = stats.linregress(x, y)
        xl = np.array([np.nanmin(x), np.nanmax(x)])
        ax.plot(xl, lr.slope * xl + lr.intercept, '-', color=color, lw=2)
        return (f'{label}: r={lr.rvalue:.3f}, p={lr.pvalue:.2e}, '
                f'slope={lr.slope:.4f}')
    return f'{label}: n<3'


def permute_y(y, rng):
    idx = rng.random(y.shape).argsort(-1)
    return np.take_along_axis(y, idx, axis=-1)


def shift_y(y, rng):
    shifts = rng.integers(0, 25, size=y.shape[:2])
    rolled_idx = (np.arange(25) - shifts[:, :, None]) % 25
    return np.take_along_axis(y, rolled_idx.astype(np.intp), axis=-1)


#%% 1. 加载、拟合、保存

cache = {}
all_dfs = []
for area in brain_areas:
    area_dir = ot.Join(datapath, area)
    rsp = np.load(ot.Join(area_dir, 'avr_rsp.npy'))
    info = pd.read_csv(ot.Join(area_dir, 'cell_site_info.csv'))
    n_cell = rsp.shape[0]

    rsp_hz = rsp / WINDOW_S
    raw, slope, r2, y = fit_raw_anchored_slope(rsp_hz=rsp_hz)
    cache[area] = {'raw': raw, 'y': y, 'rsp_hz': rsp_hz}

    rows = []
    for img in range(N_IMG):
        rows.append(pd.DataFrame({
            'area': area,
            'cell_idx': np.arange(n_cell),
            'img_id': img,
            'is_ani': img < 20,
            'raw_rsp': raw[:, img],
            'slope': slope[:, img],
            'r2': r2[:, img],
            'n_pt': 25,
        }))
    df = pd.concat(rows, ignore_index=True)
    df = df.merge(
        info[['global_idx', 'site_name', 'dprime_face', 'dprime_body']],
        left_on='cell_idx', right_index=True, how='left',
    )

    out_dir = ot.Join(savepath, area)
    ot.Mkdir(out_dir)
    df.to_csv(ot.Join(out_dir, 'slope_by_image.csv'), index=False)
    all_dfs.append(df)
    print(f'{area}: {n_cell} cells x {N_IMG} imgs -> {len(df)} rows')

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv(ot.Join(savepath, 'slope_by_image_all.csv'), index=False)
print(f'saved all: {len(df_all)} rows -> {ot.Join(savepath, "slope_by_image_all.csv")}')

#%% 2. 拟合优度 R² 统计（ani / inani 分开）

ot.Mkdir(FIG_DIR)

for area in plot_areas:
    sub = df_all[df_all['area'] == area]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, is_ani, label, color in zip(
        axes, [True, False], ['Ani', 'Inani'], [COLOR_ANI, COLOR_INANI],
    ):
        vals = sub.loc[sub['is_ani'] == is_ani, 'r2'].to_numpy()
        ax.hist(vals, bins=50, range=(0, 1), color=color, alpha=0.75, edgecolor='white')
        med = np.nanmedian(vals)
        ax.axvline(med, color='k', ls='--', lw=1.2, label=f'median={med:.2f}')
        ax.set_xlabel(r'$R^2$  (shuffle linear fit)')
        ax.set_ylabel('Count  (neuron × image)')
        ax.set_title(f'{label}  n={len(vals)}')
        ax.legend(fontsize=8)

    fig.suptitle(
        f'{area}  slope fit goodness — how linearly response drops with shuffle',
        fontsize=11,
    )
    fig.tight_layout()
    out = ot.Join(FIG_DIR, f'r2_fit_quality_{area}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {out}')

#%% 3. raw_rsp vs slope 散点 + 回归

ot.Mkdir(FIG_DIR)

for area in plot_areas:
    sub = df_all[df_all['area'] == area]
    fig, ax = plt.subplots(figsize=(6, 5))
    stat_lines = []
    for is_ani, color, label in [(True, COLOR_ANI, 'Ani'), (False, COLOR_INANI, 'Inani')]:
        stat_lines.append(plot_scatter_regress(ax, sub, 'raw_rsp', 'slope', is_ani, color, label))
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Raw firing rate (Hz)')
    ax.set_ylabel('Slope (Hz / shuffle level)')
    ax.set_title(f'{area}  raw vs slope  (points = neuron × image)')
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ANI,
               markersize=8, label='Ani'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INANI,
               markersize=8, label='Inani'),
    ], fontsize=9)
    ax.text(0.02, 0.98, '\n'.join(stat_lines), transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    out = ot.Join(FIG_DIR, f'raw_vs_slope_{area}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {out}')

#%% 3b. raw_rsp vs slope — KDE + 回归

ot.Mkdir(FIG_DIR)

for area in plot_areas:
    sub = df_all[df_all['area'] == area]
    fig, ax = plt.subplots(figsize=(6, 5))
    stat_lines = []
    for is_ani, color, label in [(True, COLOR_ANI, 'Ani'), (False, COLOR_INANI, 'Inani')]:
        stat_lines.append(plot_kde_regress(ax, sub, 'raw_rsp', 'slope', is_ani, color, label))
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Raw firing rate (Hz)')
    ax.set_ylabel('Slope (Hz / shuffle level)')
    ax.set_title(f'{area}  raw vs slope  KDE  (points = neuron × image)')
    ax.legend(fontsize=9)
    ax.text(0.02, 0.98, '\n'.join(stat_lines), transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    out = ot.Join(FIG_DIR, f'raw_vs_slope_kde_{area}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {out}')

#%% 4. raw_rsp vs R² 散点 + 回归

ot.Mkdir(FIG_DIR)

for area in plot_areas:
    sub = df_all[df_all['area'] == area]
    fig, ax = plt.subplots(figsize=(6, 5))
    stat_lines = []
    for is_ani, color, label in [(True, COLOR_ANI, 'Ani'), (False, COLOR_INANI, 'Inani')]:
        stat_lines.append(plot_scatter_regress(ax, sub, 'raw_rsp', 'r2', is_ani, color, label))
    ax.set_xlabel('Raw firing rate (Hz)')
    ax.set_ylabel(r'$R^2$  (shuffle linear fit)')
    ax.set_title(f'{area}  raw vs $R^2$  (points = neuron × image)')
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ANI,
               markersize=8, label='Ani'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INANI,
               markersize=8, label='Inani'),
    ], fontsize=9)
    ax.text(0.02, 0.98, '\n'.join(stat_lines), transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    out = ot.Join(FIG_DIR, f'raw_vs_r2_{area}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {out}')

#%% 4b. raw_rsp vs R² — KDE + 回归

ot.Mkdir(FIG_DIR)

for area in plot_areas:
    sub = df_all[df_all['area'] == area]
    fig, ax = plt.subplots(figsize=(6, 5))
    stat_lines = []
    for is_ani, color, label in [(True, COLOR_ANI, 'Ani'), (False, COLOR_INANI, 'Inani')]:
        stat_lines.append(plot_kde_regress(ax, sub, 'raw_rsp', 'r2', is_ani, color, label))
    ax.set_xlabel('Raw firing rate (Hz)')
    ax.set_ylabel(r'$R^2$  (shuffle linear fit)')
    ax.set_title(f'{area}  raw vs $R^2$  KDE  (points = neuron × image)')
    ax.legend(fontsize=9)
    ax.text(0.02, 0.98, '\n'.join(stat_lines), transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    out = ot.Join(FIG_DIR, f'raw_vs_r2_kde_{area}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {out}')

#%% 5. Control — 标签置换 + 循环移位

ot.Mkdir(FIG_DIR)

for area in plot_areas:
    area_control_rows = []
    raw = cache[area]['raw']
    y = cache[area]['y']
    _, slope_obs, r2_obs, _ = fit_raw_anchored_slope(y=y, raw=raw)
    rng = np.random.default_rng(CONTROL_SEED)

    obs = {
        'ani': group_metrics(raw, slope_obs, r2_obs, ANI_IMG),
        'inani': group_metrics(raw, slope_obs, r2_obs, INANI_IMG),
    }

    null_store = {
        'perm': {k: {m: [] for m in obs['ani']} for k in ('ani', 'inani')},
        'shift': {k: {m: [] for m in obs['ani']} for k in ('ani', 'inani')},
    }

    for _ in range(N_PERM):
        y_null = permute_y(y, rng)
        _, sl, r2, _ = fit_raw_anchored_slope(y=y_null, raw=raw)
        for key, imgs in [('ani', ANI_IMG), ('inani', INANI_IMG)]:
            m = group_metrics(raw, sl, r2, imgs)
            for mk, mv in m.items():
                null_store['perm'][key][mk].append(mv)

    for _ in range(N_SHIFT):
        y_null = shift_y(y, rng)
        _, sl, r2, _ = fit_raw_anchored_slope(y=y_null, raw=raw)
        for key, imgs in [('ani', ANI_IMG), ('inani', INANI_IMG)]:
            m = group_metrics(raw, sl, r2, imgs)
            for mk, mv in m.items():
                null_store['shift'][key][mk].append(mv)

    for ctrl_type in ('perm', 'shift'):
        for key, is_ani in [('ani', True), ('inani', False)]:
            for metric, obs_val in obs[key].items():
                null_vals = null_store[ctrl_type][key][metric]
                use_abs = metric.startswith('corr')
                pval = null_p_value(null_vals, obs_val, use_abs=use_abs)
                area_control_rows.append({
                    'area': area,
                    'is_ani': is_ani,
                    'metric': metric,
                    'observed': obs_val,
                    'null_median': float(np.median(null_vals)),
                    'p_value': pval,
                    'control_type': ctrl_type,
                })
                print(f'{area} {ctrl_type} {key} {metric}: obs={obs_val:.4f}  '
                      f'null_med={np.median(null_vals):.4f}  p={pval:.4f}')

    # --- control null histograms (2×2) ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (0, 0, 'ani', 'median_abs_slope', r'$|\mathrm{slope}|$ median'),
        (0, 1, 'inani', 'median_abs_slope', r'$|\mathrm{slope}|$ median'),
        (1, 0, 'ani', 'median_r2', r'$R^2$ median'),
        (1, 1, 'inani', 'median_r2', r'$R^2$ median'),
    ]
    for row, col, key, metric, xlabel in panels:
        ax = axes[row, col]
        obs_val = obs[key][metric]
        color = COLOR_ANI if key == 'ani' else COLOR_INANI
        for ctrl_type, alpha, lbl in [('perm', 0.55, 'perm'), ('shift', 0.55, 'shift')]:
            ax.hist(null_store[ctrl_type][key][metric], bins=40, color='0.7',
                    alpha=alpha, edgecolor='white', label=f'null {lbl}')
        ax.axvline(obs_val, color=color, lw=2.5, label=f'obs={obs_val:.3f}')
        p_perm = null_p_value(null_store['perm'][key][metric], obs_val)
        p_shift = null_p_value(null_store['shift'][key][metric], obs_val)
        ax.set_title(f'{key.capitalize()}  p_perm={p_perm:.3f}  p_shift={p_shift:.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)
    fig.suptitle(f'{area}  shuffle control — obs vs null', fontsize=11)
    fig.tight_layout()
    out = ot.Join(FIG_DIR, f'control_null_{area}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {out}')

    # --- control correlation ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    corr_panels = [
        (0, 0, 'ani', 'corr_raw_slope', 'corr(raw, slope)'),
        (0, 1, 'inani', 'corr_raw_slope', 'corr(raw, slope)'),
        (1, 0, 'ani', 'corr_raw_r2', 'corr(raw, R²)'),
        (1, 1, 'inani', 'corr_raw_r2', 'corr(raw, R²)'),
    ]
    for row, col, key, metric, xlabel in corr_panels:
        ax = axes[row, col]
        obs_val = obs[key][metric]
        color = COLOR_ANI if key == 'ani' else COLOR_INANI
        for ctrl_type, lbl in [('perm', 'perm'), ('shift', 'shift')]:
            ax.hist(null_store[ctrl_type][key][metric], bins=40, color='0.7',
                    alpha=0.55, edgecolor='white', label=f'null {lbl}')
        ax.axvline(obs_val, color=color, lw=2.5, label=f'obs={obs_val:.3f}')
        p_perm = null_p_value(null_store['perm'][key][metric], obs_val, use_abs=True)
        p_shift = null_p_value(null_store['shift'][key][metric], obs_val, use_abs=True)
        ax.set_title(f'{key.capitalize()}  p_perm={p_perm:.3f}  p_shift={p_shift:.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)
    fig.suptitle(f'{area}  raw correlation control', fontsize=11)
    fig.tight_layout()
    out = ot.Join(FIG_DIR, f'control_corr_{area}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {out}')

    pd.DataFrame(area_control_rows).to_csv(
        ot.Join(savepath, area, 'control_summary.csv'), index=False,
    )
 