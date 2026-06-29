#%% Directory and imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import OS_Tools as ot

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


datapath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\decay_index'

brain_areas = ['ML', 'MSB', 'AL', 'ASB']
plot_areas = ['ML', 'MSB', 'AL', 'ASB']
WINDOW_S = 0.17
N_REPEAT = 5
N_SHUF = 5
N_IMG = 40
N_PERM = 200
N_RAW_BIN = 5
MIN_VALID_IMG = 6
RAW_MIN_HZ = 5
CONTROL_SEED = 0
FIG_DIR = ot.Join(savepath, 'figures', 'Decay_Index')
COLOR_ANI = '#c0392b'
COLOR_INANI = '#2980b9'
COLOR_ALL = '#2c3e50'
ANI_IMG = np.arange(20)
INANI_IMG = np.arange(20, 40)
IMAGE_SCOPES = [
    ('all', 'All images', None, COLOR_ALL),
    ('ani', 'Ani images', True, COLOR_ANI),
    ('inani', 'Inani images', False, COLOR_INANI),
]

# shuffle level for each of 25 points: repeat0 shuf0-4, repeat1 shuf0-4, ...
X_SHUF = np.tile(np.arange(N_SHUF, dtype=float), N_REPEAT)
EPS = 1e-12


def progress_iter(iterable, desc, leave=False):
    """Use tqdm when available, otherwise print a simple progress label."""
    if tqdm is not None:
        return tqdm(iterable, desc=desc, leave=leave)
    print(desc)
    return iterable


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
    return raw, y, rsp_hz, info


def fit_decay_beta(raw, y):
    """Fit response = raw_rsp * (1 - beta * shuffle_level) for cell x image."""
    x = X_SHUF[None, None, :]
    valid = np.isfinite(y)
    raw_valid = np.isfinite(raw) & (raw > RAW_MIN_HZ)
    ok_fit = raw_valid & (valid.sum(-1) >= 3)

    predictor = raw[..., None] * x
    target = raw[..., None] - y

    num = np.where(valid, predictor * target, 0.0).sum(-1)
    den = np.where(valid, predictor ** 2, 0.0).sum(-1)
    ok_fit = ok_fit & (den > EPS)

    beta = np.full(raw.shape, np.nan, dtype=float)
    beta[ok_fit] = num[ok_fit] / den[ok_fit]
    slope_hz = -raw * beta

    y_hat = raw[..., None] * (1.0 - beta[..., None] * x)
    n = valid.sum(-1).astype(float)
    sy = np.where(valid, y, 0.0).sum(-1)
    y_mean = np.divide(sy, n, out=np.full_like(sy, np.nan), where=n > 0)
    ss_res = np.where(valid, (y - y_hat) ** 2, 0.0).sum(-1)
    ss_tot = np.where(valid, (y - y_mean[..., None]) ** 2, 0.0).sum(-1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, EPS)
    r2[~ok_fit] = np.nan

    return {
        'beta': beta,
        'slope_hz_per_level': slope_hz,
        'beta_signed': -beta,
        'decay_beta': beta,
        'r2': r2,
        'n_pt': n.astype(int),
        'raw_valid': raw_valid,
    }


def build_beta_frame(area, raw, fit, info):
    """Build the cell x image beta dataframe for one area."""
    n_cell = raw.shape[0]
    rows = []
    for img in range(N_IMG):
        rows.append(pd.DataFrame({
            'area': area,
            'cell_idx': np.arange(n_cell),
            'img_id': img,
            'is_ani': img < 20,
            'raw_rsp': raw[:, img],
            'beta': fit['beta'][:, img],
            'slope_hz_per_level': fit['slope_hz_per_level'][:, img],
            'beta_signed': fit['beta_signed'][:, img],
            'decay_beta': fit['decay_beta'][:, img],
            'r2': fit['r2'][:, img],
            'n_pt': fit['n_pt'][:, img],
            'raw_valid': fit['raw_valid'][:, img],
        }))
    df = pd.concat(rows, ignore_index=True)

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


def finite_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def corr_stats(x, y, prefix):
    """Return Pearson and Spearman correlation stats for finite pairs."""
    x, y = finite_pair(x, y)
    out = {
        f'{prefix}_n': int(len(x)),
        f'{prefix}_pearson_r': np.nan,
        f'{prefix}_pearson_p': np.nan,
        f'{prefix}_spearman_r': np.nan,
        f'{prefix}_spearman_p': np.nan,
    }
    if len(x) > 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        out.update({
            f'{prefix}_pearson_r': float(pr.statistic),
            f'{prefix}_pearson_p': float(pr.pvalue),
            f'{prefix}_spearman_r': float(sr.statistic),
            f'{prefix}_spearman_p': float(sr.pvalue),
        })
    return out


def filter_image_scope(df, scope_flag):
    """Return rows for all images, ani only, or inani only."""
    if scope_flag is None:
        return df
    return df[df['is_ani'] == scope_flag]


def nan_iqr(vals):
    vals = np.asarray(vals, dtype=float)
    if np.isfinite(vals).sum() == 0:
        return np.nan
    return float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))


def nan_mad(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    med = np.nanmedian(vals)
    return float(np.nanmedian(np.abs(vals - med)))


def add_within_cell_residuals(df):
    """Add beta residuals after removing each cell's linear raw-response effect."""
    df = df.copy()
    df['beta_signed_resid_raw'] = np.nan
    df['decay_beta_resid_raw'] = np.nan
    df['raw_bin_within_cell'] = np.nan

    for _, sub in df.groupby(['area', 'cell_idx'], sort=False):
        idx = sub.index.to_numpy()
        valid = (
            sub['raw_valid'].to_numpy(dtype=bool)
            & np.isfinite(sub['raw_rsp'].to_numpy(dtype=float))
            & np.isfinite(sub['beta_signed'].to_numpy(dtype=float))
        )
        if valid.sum() < 3:
            continue

        v_idx = idx[valid]
        raw = sub.loc[v_idx, 'raw_rsp'].to_numpy(dtype=float)
        beta = sub.loc[v_idx, 'beta_signed'].to_numpy(dtype=float)

        if np.nanstd(raw) > 0:
            lr = stats.linregress(raw, beta)
            resid = beta - (lr.intercept + lr.slope * raw)
        else:
            resid = beta - np.nanmean(beta)

        df.loc[v_idx, 'beta_signed_resid_raw'] = resid
        df.loc[v_idx, 'decay_beta_resid_raw'] = -resid

        if len(v_idx) >= N_RAW_BIN:
            order = np.argsort(raw, kind='mergesort')
            bins = np.floor(np.arange(len(order)) * N_RAW_BIN / len(order)).astype(int)
            bins = np.clip(bins, 0, N_RAW_BIN - 1)
            df.loc[v_idx[order], 'raw_bin_within_cell'] = bins

    return df


def summarize_one_cell(sub):
    valid = sub['raw_valid'] & np.isfinite(sub['beta_signed'])
    v = sub.loc[valid]
    out = {
        'area': sub['area'].iloc[0],
        'cell_idx': int(sub['cell_idx'].iloc[0]),
        'n_valid_img': int(len(v)),
        'raw_mean': float(v['raw_rsp'].mean()) if len(v) else np.nan,
        'raw_median': float(v['raw_rsp'].median()) if len(v) else np.nan,
        'raw_std': float(v['raw_rsp'].std(ddof=1)) if len(v) > 1 else np.nan,
        'beta_signed_mean': float(v['beta_signed'].mean()) if len(v) else np.nan,
        'beta_signed_median': float(v['beta_signed'].median()) if len(v) else np.nan,
        'beta_signed_std': float(v['beta_signed'].std(ddof=1)) if len(v) > 1 else np.nan,
        'beta_signed_iqr': nan_iqr(v['beta_signed']) if len(v) else np.nan,
        'beta_signed_mad': nan_mad(v['beta_signed']) if len(v) else np.nan,
        'decay_beta_mean': float(v['decay_beta'].mean()) if len(v) else np.nan,
        'decay_beta_median': float(v['decay_beta'].median()) if len(v) else np.nan,
        'decay_beta_std': float(v['decay_beta'].std(ddof=1)) if len(v) > 1 else np.nan,
        'beta_resid_std': float(v['beta_signed_resid_raw'].std(ddof=1)) if len(v) > 1 else np.nan,
        'beta_resid_iqr': nan_iqr(v['beta_signed_resid_raw']) if len(v) else np.nan,
        'beta_resid_mad': nan_mad(v['beta_signed_resid_raw']) if len(v) else np.nan,
    }

    ani = v[v['is_ani']]
    inani = v[~v['is_ani']]
    out['decay_beta_ani_mean'] = float(ani['decay_beta'].mean()) if len(ani) else np.nan
    out['decay_beta_inani_mean'] = float(inani['decay_beta'].mean()) if len(inani) else np.nan
    out['decay_beta_ani_minus_inani'] = out['decay_beta_ani_mean'] - out['decay_beta_inani_mean']

    out.update(corr_stats(v['raw_rsp'], v['beta_signed'], 'within_raw_beta_signed'))
    out.update(corr_stats(v['raw_rsp'], v['decay_beta'], 'within_raw_decay_beta'))

    meta_cols = ['global_idx', 'site_name', 'dprime_face', 'dprime_body']
    for col in meta_cols:
        if col in sub.columns:
            out[col] = sub[col].iloc[0]
    return out


def summarize_cells(df):
    rows = []
    for _, sub in df.groupby(['area', 'cell_idx'], sort=False):
        rows.append(summarize_one_cell(sub))
    return pd.DataFrame(rows)


def raw_bin_summary_for_df(df):
    """Build raw-response bins within each cell for the provided image scope."""
    work = df[df['raw_valid'] & np.isfinite(df['raw_rsp']) & np.isfinite(df['decay_beta'])].copy()
    work['raw_bin_scope'] = np.nan

    for _, sub in work.groupby(['area', 'cell_idx'], sort=False):
        if len(sub) < N_RAW_BIN:
            continue
        raw = sub['raw_rsp'].to_numpy(dtype=float)
        order = np.argsort(raw, kind='mergesort')
        bins = np.floor(np.arange(len(order)) * N_RAW_BIN / len(order)).astype(int)
        bins = np.clip(bins, 0, N_RAW_BIN - 1)
        work.loc[sub.index.to_numpy()[order], 'raw_bin_scope'] = bins

    return (
        work[np.isfinite(work['raw_bin_scope'])]
        .groupby(['area', 'raw_bin_scope'], as_index=False)
        .agg(
            n=('decay_beta', 'count'),
            raw_rsp_mean=('raw_rsp', 'mean'),
            decay_beta_mean=('decay_beta', 'mean'),
            decay_beta_sem=('decay_beta', 'sem'),
            beta_signed_mean=('beta_signed', 'mean'),
            beta_signed_sem=('beta_signed', 'sem'),
        )
    )


def centered_cell_arrays(sub, y_col='beta_signed'):
    xs = []
    ys = []
    for _, g in sub.groupby(['area', 'cell_idx'], sort=False):
        valid = g['raw_valid'] & np.isfinite(g['raw_rsp']) & np.isfinite(g[y_col])
        v = g.loc[valid]
        if len(v) < MIN_VALID_IMG:
            continue
        x = v['raw_rsp'].to_numpy(dtype=float)
        y = v[y_col].to_numpy(dtype=float)
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            continue
        xs.append(x - x.mean())
        ys.append(y - y.mean())

    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def centered_cell_corr(sub, y_col='beta_signed'):
    x, y = centered_cell_arrays(sub, y_col=y_col)
    if len(x) <= 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(stats.pearsonr(x, y).statistic)


def per_cell_raw_beta_corr_control(sub, area_label, scope_name, rng, y_col='decay_beta'):
    """Correlate raw and beta within each cell, with within-cell shuffled controls."""
    obs_rows = []
    cell_items = []

    for _, g in sub.groupby(['area', 'cell_idx'], sort=False):
        valid = g['raw_valid'] & np.isfinite(g['raw_rsp']) & np.isfinite(g[y_col])
        v = g.loc[valid]
        if len(v) < MIN_VALID_IMG:
            continue

        raw = v['raw_rsp'].to_numpy(dtype=float)
        beta = v[y_col].to_numpy(dtype=float)
        if np.nanstd(raw) == 0 or np.nanstd(beta) == 0:
            continue

        r_obs = float(stats.pearsonr(raw, beta).statistic)
        cell_area = v['area'].iloc[0]
        cell_idx = int(v['cell_idx'].iloc[0])
        obs_rows.append({
            'area': area_label,
            'cell_area': cell_area,
            'cell_idx': cell_idx,
            'image_scope': scope_name,
            'n_img': int(len(v)),
            'raw_beta_r': r_obs,
        })
        cell_items.append((raw, beta))

    obs_df = pd.DataFrame(obs_rows)
    null_by_perm = []
    null_pooled = []

    perm_desc = f'perm raw-beta corr {area_label} {scope_name}'
    for _ in progress_iter(range(N_PERM), perm_desc, leave=False):
        perm_rs = []
        for raw, beta in cell_items:
            r_null = float(stats.pearsonr(raw, rng.permutation(beta)).statistic)
            perm_rs.append(r_null)
            null_pooled.append(r_null)
        if perm_rs:
            null_by_perm.append(float(np.nanmedian(perm_rs)))

    obs_median = float(obs_df['raw_beta_r'].median()) if len(obs_df) else np.nan
    null_by_perm = np.asarray(null_by_perm, dtype=float)
    null_pooled = np.asarray(null_pooled, dtype=float)
    if len(null_by_perm):
        p_value = float(np.mean(np.abs(null_by_perm) >= abs(obs_median)))
        null_median = float(np.nanmedian(null_by_perm))
    else:
        p_value = np.nan
        null_median = np.nan

    summary = {
        'area': area_label,
        'image_scope': scope_name,
        'metric': f'per_cell_corr_raw_{y_col}',
        'n_cell': int(len(obs_df)),
        'observed_median': obs_median,
        'observed_mean': float(obs_df['raw_beta_r'].mean()) if len(obs_df) else np.nan,
        'control_median_of_medians': null_median,
        'p_value_two_sided': p_value,
        'n_perm': N_PERM,
    }
    return obs_df, null_pooled, null_by_perm, summary


def plot_scatter_regress_single(ax, sub, xcol, ycol, color, label):
    m = sub['raw_valid']
    x = sub.loc[m, xcol].to_numpy(dtype=float)
    y = sub.loc[m, ycol].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    ax.scatter(x, y, s=6, color=color, alpha=0.25, edgecolors='none', rasterized=True)
    if len(x) > 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        lr = stats.linregress(x, y)
        xl = np.array([np.nanmin(x), np.nanmax(x)])
        ax.plot(xl, lr.slope * xl + lr.intercept, '-', color=color, lw=2)
        return f'{label}: r={lr.rvalue:.3f}, p={lr.pvalue:.2e}, slope={lr.slope:.4f}'
    return f'{label}: n<3'


#%% 1. Load, fit beta, save cell x image tables

ot.Mkdir(savepath)
ot.Mkdir(FIG_DIR)

cache = {}
all_dfs = []
for area in brain_areas:
    raw, y, rsp_hz, info = load_area_response(area)
    fit = fit_decay_beta(raw, y)
    cache[area] = {'raw': raw, 'y': y, 'rsp_hz': rsp_hz, 'fit': fit}

    df = build_beta_frame(area, raw, fit, info)
    out_dir = ot.Join(savepath, area)
    ot.Mkdir(out_dir)
    df.to_csv(ot.Join(out_dir, 'decay_beta_by_image.csv'), index=False)
    all_dfs.append(df)
    print(f'{area}: {raw.shape[0]} cells x {N_IMG} imgs -> {len(df)} rows')

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = add_within_cell_residuals(df_all)
df_all.to_csv(ot.Join(savepath, 'decay_beta_by_image_all.csv'), index=False)
print(f'saved all: {len(df_all)} rows -> {ot.Join(savepath, "decay_beta_by_image_all.csv")}')


#%% 2. Fit quality and beta QC plots

for area in plot_areas:
    area_df = df_all[df_all['area'] == area]

    for scope_name, scope_label, scope_flag, scope_color in IMAGE_SCOPES:
        sub = filter_image_scope(area_df, scope_flag)

        fig, ax = plt.subplots(figsize=(5, 4))
        vals = sub.loc[sub['raw_valid'], 'r2'].to_numpy()
        ax.hist(vals[np.isfinite(vals)], bins=50, range=(0, 1), color=scope_color, alpha=0.75,
                edgecolor='white')
        med = np.nanmedian(vals)
        ax.axvline(med, color='k', ls='--', lw=1.2, label=f'median={med:.2f}')
        ax.set_xlabel('R2  (one-parameter beta fit)')
        ax.set_ylabel('Count  (neuron x image)')
        ax.set_title(f'{area}  {scope_label}  n={np.isfinite(vals).sum()}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = ot.Join(FIG_DIR, f'r2_fit_quality_{area}_{scope_name}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'saved: {out}')

        fig, ax = plt.subplots(figsize=(6, 5))
        stat_line = plot_scatter_regress_single(
            ax, sub, 'raw_rsp', 'decay_beta', scope_color, scope_label,
        )
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Raw firing rate (Hz)')
        ax.set_ylabel('Beta  (decay per shuffle level)')
        ax.set_title(f'{area}  {scope_label}  raw response vs beta')
        ax.text(0.02, 0.98, stat_line, transform=ax.transAxes,
                fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        fig.tight_layout()
        out = ot.Join(FIG_DIR, f'raw_vs_beta_{area}_{scope_name}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'saved: {out}')

        fig, ax = plt.subplots(figsize=(6, 5))
        stat_line = plot_scatter_regress_single(
            ax, sub, 'decay_beta', 'r2', scope_color, scope_label,
        )
        ax.set_xlabel('Beta  (decay per shuffle level)')
        ax.set_ylabel('R2  (one-parameter beta fit)')
        ax.set_title(f'{area}  {scope_label}  beta vs fit quality')
        ax.text(0.02, 0.98, stat_line, transform=ax.transAxes,
                fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        fig.tight_layout()
        out = ot.Join(FIG_DIR, f'r2_vs_beta_{area}_{scope_name}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'saved: {out}')

        fig, ax = plt.subplots(figsize=(5, 4))
        vals = sub.loc[sub['raw_valid'], 'decay_beta'].to_numpy()
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=60, color=scope_color, alpha=0.75, edgecolor='white')
        med = np.nanmedian(vals)
        ax.axvline(med, color='k', ls='--', lw=1.2, label=f'median={med:.3f}')
        ax.axvline(0, color='gray', ls=':', lw=1.0)
        ax.set_xlabel('Beta  (decay per shuffle level)')
        ax.set_ylabel('Count  (neuron x image)')
        ax.set_title(f'{area}  {scope_label}  n={len(vals)}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = ot.Join(FIG_DIR, f'beta_distribution_{area}_{scope_name}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'saved: {out}')


#%% 3. Within-cell beta differences after accounting for raw response

df_cell_by_scope = {}
raw_bin_summary_by_scope = {}

for scope_name, scope_label, scope_flag, _ in progress_iter(
    IMAGE_SCOPES, 'raw-beta binding scopes', leave=True,
):
    scope_df = filter_image_scope(df_all, scope_flag)
    df_cell_scope = summarize_cells(scope_df)
    raw_bin_scope = raw_bin_summary_for_df(scope_df)

    df_cell_by_scope[scope_name] = df_cell_scope
    raw_bin_summary_by_scope[scope_name] = raw_bin_scope

    df_cell_scope.to_csv(
        ot.Join(savepath, f'decay_beta_cell_summary_{scope_name}.csv'), index=False,
    )
    raw_bin_scope.to_csv(
        ot.Join(savepath, f'raw_bin_beta_summary_{scope_name}.csv'), index=False,
    )

    for area in brain_areas:
        out_dir = ot.Join(savepath, area)
        df_cell_scope[df_cell_scope['area'] == area].to_csv(
            ot.Join(out_dir, f'decay_beta_cell_summary_{scope_name}.csv'), index=False,
        )

    print(f'saved {scope_name} cell summary and raw-bin summary')

df_cell = df_cell_by_scope['all']

for area in plot_areas:
    for scope_name, scope_label, _, scope_color in IMAGE_SCOPES:
        sub_cell = df_cell_by_scope[scope_name]
        sub_cell = sub_cell[sub_cell['area'] == area]
        valid = sub_cell[sub_cell['n_valid_img'] >= MIN_VALID_IMG]

        plot_df = valid.sort_values('decay_beta_mean').reset_index(drop=True)
        x = np.arange(len(plot_df))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].errorbar(
            x,
            plot_df['decay_beta_mean'],
            yerr=plot_df['decay_beta_std'],
            fmt='o',
            ms=3,
            lw=0.8,
            color=scope_color,
            ecolor='0.7',
            capsize=0,
        )
        axes[0].axhline(0, color='gray', ls='--', lw=0.8)
        axes[0].set_xlabel('Cells sorted by mean beta')
        axes[0].set_ylabel('Beta mean +/- std across images')
        axes[0].set_title('Each cell: image-to-image beta spread')

        axes[1].scatter(
            valid['decay_beta_mean'],
            valid['decay_beta_std'],
            s=14,
            color=scope_color,
            alpha=0.6,
            edgecolors='none',
            rasterized=True,
        )
        axes[1].set_xlabel('Mean beta across images')
        axes[1].set_ylabel('Std beta across images')
        axes[1].set_title('Cells with larger std vary more by image')

        fig.suptitle(f'{area}  {scope_label}  cell-level beta mean and std', fontsize=11)
        fig.tight_layout()
        out = ot.Join(FIG_DIR, f'cell_beta_mean_std_{area}_{scope_name}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'saved: {out}')

        sub_bin = raw_bin_summary_by_scope[scope_name]
        sub_bin = sub_bin[sub_bin['area'] == area]
        if len(sub_bin):
            fig, ax = plt.subplots(figsize=(5, 4))
            x = sub_bin['raw_bin_scope'].to_numpy(dtype=float)
            y = sub_bin['decay_beta_mean'].to_numpy(dtype=float)
            sem = sub_bin['decay_beta_sem'].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=sem, marker='o', color=scope_color, capsize=3)
            ax.axhline(0, color='gray', ls='--', lw=0.8)
            ax.set_xticks(np.arange(N_RAW_BIN))
            ax.set_xlabel('Within-cell raw response bin')
            ax.set_ylabel('Mean beta')
            ax.set_title(f'{area}  {scope_label}  raw-bin beta trend')
            fig.tight_layout()
            out = ot.Join(FIG_DIR, f'raw_bin_beta_{area}_{scope_name}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.show()
            print(f'saved: {out}')


#%% 4. Raw response and beta binding tests

rng = np.random.default_rng(CONTROL_SEED)
binding_rows = []
cell_corr_rows = []
cell_corr_plot_store = {}

for scope_name, scope_label, scope_flag, _ in IMAGE_SCOPES:
    scope_all = filter_image_scope(df_all, scope_flag)
    scope_cell = df_cell_by_scope[scope_name]
    area_items = [('all_areas', scope_all)] + [
        (area, scope_all[scope_all['area'] == area]) for area in plot_areas
    ]

    for area_label, sub in progress_iter(area_items, f'areas {scope_name}', leave=False):
        valid = sub[sub['raw_valid']]
        row = {
            'area': area_label,
            'image_scope': scope_name,
            'image_scope_label': scope_label,
            'n_point': int(np.isfinite(valid['decay_beta']).sum()),
            'n_cell': int(valid[['area', 'cell_idx']].drop_duplicates().shape[0]),
        }
        row.update(corr_stats(valid['raw_rsp'], valid['decay_beta'], 'pooled_raw_beta'))
        row.update(corr_stats(valid['raw_rsp'], valid['beta_signed'], 'pooled_raw_beta_signed'))

        if area_label == 'all_areas':
            cell_mask = np.ones(len(scope_cell), dtype=bool)
        else:
            cell_mask = scope_cell['area'] == area_label
        cell_corr = scope_cell.loc[
            cell_mask & (scope_cell['n_valid_img'] >= MIN_VALID_IMG),
            'within_raw_decay_beta_pearson_r',
        ].dropna()
        row['within_cell_corr_mean'] = float(cell_corr.mean()) if len(cell_corr) else np.nan
        row['within_cell_corr_median'] = float(cell_corr.median()) if len(cell_corr) else np.nan
        row['within_cell_corr_ttest_p'] = (
            float(stats.ttest_1samp(cell_corr, 0.0).pvalue) if len(cell_corr) > 1 else np.nan
        )
        row['within_cell_centered_corr_raw_beta'] = centered_cell_corr(sub, 'decay_beta')

        obs_df, null_pooled, null_by_perm, ctrl_summary = per_cell_raw_beta_corr_control(
            sub, area_label, scope_name, rng, y_col='decay_beta',
        )
        row.update({
            'per_cell_corr_median': ctrl_summary['observed_median'],
            'per_cell_corr_mean': ctrl_summary['observed_mean'],
            'control_median_of_medians': ctrl_summary['control_median_of_medians'],
            'control_p_value_two_sided': ctrl_summary['p_value_two_sided'],
            'control_n_perm': ctrl_summary['n_perm'],
        })
        binding_rows.append(row)
        if len(obs_df):
            cell_corr_rows.append(obs_df)
        cell_corr_plot_store[(area_label, scope_name)] = {
            'obs': obs_df['raw_beta_r'].to_numpy(dtype=float) if len(obs_df) else np.array([]),
            'null_pooled': null_pooled,
            'null_by_perm': null_by_perm,
            'summary': ctrl_summary,
        }

df_binding = pd.DataFrame(binding_rows)
df_binding.to_csv(ot.Join(savepath, 'raw_beta_binding_summary.csv'), index=False)
print(f'saved binding summary: {ot.Join(savepath, "raw_beta_binding_summary.csv")}')

if cell_corr_rows:
    df_cell_corr = pd.concat(cell_corr_rows, ignore_index=True)
else:
    df_cell_corr = pd.DataFrame()
df_cell_corr.to_csv(ot.Join(savepath, 'per_cell_raw_beta_corr.csv'), index=False)
print(f'saved per-cell raw-beta corr: {ot.Join(savepath, "per_cell_raw_beta_corr.csv")}')

for area in plot_areas:
    for scope_name, scope_label, _, scope_color in IMAGE_SCOPES:
        plot_data = cell_corr_plot_store[(area, scope_name)]
        obs_vals = plot_data['obs']
        null_vals = plot_data['null_pooled']
        summary = plot_data['summary']
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(-1, 1, 41)
        if len(null_vals):
            ax.hist(null_vals, bins=bins, color='0.75', alpha=0.75, edgecolor='white',
                    density=True, label='Control: beta shuffled within cell')
        if len(obs_vals):
            ax.hist(obs_vals, bins=bins, color=scope_color, alpha=0.55, edgecolor='white',
                    density=True, label='Observed cells')
            ax.axvline(summary['observed_median'], color=scope_color, lw=2.5,
                       label=f'obs median={summary["observed_median"]:.3f}')
        if len(plot_data['null_by_perm']):
            ax.axvline(summary['control_median_of_medians'], color='0.25', lw=2.0, ls=':',
                       label=f'ctrl median={summary["control_median_of_medians"]:.3f}')
        ax.axvline(0, color='gray', ls='--', lw=0.8)
        p_text = summary['p_value_two_sided']
        p_label = 'nan' if not np.isfinite(p_text) else f'{p_text:.3f}'
        ax.set_xlabel('Per-cell corr(raw response across images, beta across images)')
        ax.set_ylabel('Density')
        ax.set_title(f'{area}  {scope_label}  per-cell raw-beta correlation  p={p_label}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = ot.Join(FIG_DIR, f'per_cell_raw_beta_corr_{area}_{scope_name}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'saved: {out}')


def build_neuron_beta_binding_frame(df):
    """One row per neuron x image scope: corr(raw response across images, beta)."""
    meta_cols = ['global_idx', 'site_name', 'dprime_face', 'dprime_body']
    rows = []

    for scope_name, scope_label, scope_flag, _ in IMAGE_SCOPES:
        scope_df = filter_image_scope(df, scope_flag)
        expected_n_img = N_IMG if scope_flag is None else N_IMG // 2

        for (area, cell_idx), sub in scope_df.groupby(['area', 'cell_idx'], sort=False):
            valid = sub['raw_valid'] & np.isfinite(sub['raw_rsp']) & np.isfinite(sub['decay_beta'])
            v = sub.loc[valid]
            raw = v['raw_rsp'].to_numpy(dtype=float)
            beta = v['decay_beta'].to_numpy(dtype=float)

            pearson_r = np.nan
            pearson_p = np.nan
            if len(v) >= 3 and np.nanstd(raw) > 0 and np.nanstd(beta) > 0:
                pr = stats.pearsonr(raw, beta)
                pearson_r = float(pr.statistic)
                pearson_p = float(pr.pvalue)

            row = {
                'area': area,
                'cell_idx': int(cell_idx),
                'image_scope': scope_name,
                'image_scope_label': scope_label,
                'n_img_expected': expected_n_img,
                'n_img_valid': int(len(v)),
                'raw_beta_pearson_r': pearson_r,
                'raw_beta_pearson_p': pearson_p,
                'raw_rsp_mean': float(np.nanmean(raw)) if len(v) else np.nan,
                'raw_rsp_std': float(np.nanstd(raw, ddof=1)) if len(v) > 1 else np.nan,
                'beta_mean': float(np.nanmean(beta)) if len(v) else np.nan,
                'beta_std': float(np.nanstd(beta, ddof=1)) if len(v) > 1 else np.nan,
            }

            for col in meta_cols:
                if col in sub.columns:
                    row[col] = sub[col].iloc[0]

            rows.append(row)

    return pd.DataFrame(rows)


df_neuron_beta_binding = build_neuron_beta_binding_frame(df_all)
df_neuron_beta_binding.to_csv(
    ot.Join(savepath, 'neuron_beta_binding_by_scope.csv'), index=False,
)
print(
    f'saved neuron beta binding: '
    f'{ot.Join(savepath, "neuron_beta_binding_by_scope.csv")}'
)

#%%

def build_dprime_binding_summary(df_binding):
    """Test whether neurons with larger dprime have stronger raw-beta binding."""
    work = df_binding.copy()
    work['binding_abs_r'] = work['raw_beta_pearson_r'].abs()

    rows = []
    for scope_name, scope_label, _, _ in IMAGE_SCOPES:
        scope_df = work[work['image_scope'] == scope_name]
        for area_label, area_df in [('all_areas', scope_df)] + [
            (area, scope_df[scope_df['area'] == area]) for area in plot_areas
        ]:
            for dp_col in ['dprime_face', 'dprime_body']:
                if dp_col not in area_df.columns:
                    continue
                for bind_col in ['raw_beta_pearson_r', 'binding_abs_r']:
                    x, y = finite_pair(area_df[dp_col], area_df[bind_col])
                    row = {
                        'area': area_label,
                        'image_scope': scope_name,
                        'image_scope_label': scope_label,
                        'dprime_col': dp_col,
                        'binding_col': bind_col,
                        'n_cell': int(len(x)),
                        'pearson_r': np.nan,
                        'pearson_p': np.nan,
                        'spearman_r': np.nan,
                        'spearman_p': np.nan,
                    }
                    if len(x) > 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                        pr = stats.pearsonr(x, y)
                        sr = stats.spearmanr(x, y)
                        row.update({
                            'pearson_r': float(pr.statistic),
                            'pearson_p': float(pr.pvalue),
                            'spearman_r': float(sr.statistic),
                            'spearman_p': float(sr.pvalue),
                        })
                    rows.append(row)

    return pd.DataFrame(rows)


df_dprime_binding_summary = build_dprime_binding_summary(df_neuron_beta_binding)
df_dprime_binding_summary.to_csv(
    ot.Join(savepath, 'dprime_binding_summary.csv'), index=False,
)
print(f'saved dprime-binding summary: {ot.Join(savepath, "dprime_binding_summary.csv")}')

plot_df = df_neuron_beta_binding.copy()
plot_df['binding_abs_r'] = plot_df['raw_beta_pearson_r'].abs()

for scope_name, scope_label, _, scope_color in IMAGE_SCOPES:
    scope_df = plot_df[plot_df['image_scope'] == scope_name]
    for area in plot_areas:
        area_df = scope_df[scope_df['area'] == area]
        for dp_col, dp_label in [('dprime_face', 'Face dprime'), ('dprime_body', 'Body dprime')]:
            if dp_col not in area_df.columns:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, bind_col, bind_label in zip(
                axes,
                ['raw_beta_pearson_r', 'binding_abs_r'],
                ['Signed binding r', 'Binding strength |r|'],
            ):
                x, y = finite_pair(area_df[dp_col], area_df[bind_col])
                ax.scatter(x, y, s=14, color=scope_color, alpha=0.5,
                           edgecolors='none', rasterized=True)
                if len(x) > 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                    lr = stats.linregress(x, y)
                    xl = np.array([np.nanmin(x), np.nanmax(x)])
                    ax.plot(xl, lr.intercept + lr.slope * xl, color='k', lw=1.5)
                    stat_text = f'r={lr.rvalue:.3f}, p={lr.pvalue:.2e}, n={len(x)}'
                else:
                    stat_text = f'n={len(x)}'
                ax.axhline(0, color='gray', ls='--', lw=0.8)
                ax.set_xlabel(dp_label)
                ax.set_ylabel(bind_label)
                ax.set_title(stat_text)

            fig.suptitle(f'{area}  {scope_label}: dprime vs raw-beta binding', fontsize=11)
            fig.tight_layout()
            out = ot.Join(FIG_DIR, f'dprime_vs_binding_{area}_{dp_col}_{scope_name}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.show()
            print(f'saved: {out}')
