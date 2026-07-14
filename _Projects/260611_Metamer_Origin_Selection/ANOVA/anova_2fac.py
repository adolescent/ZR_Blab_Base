'''
Per-neuron 2-way ANOVA: image_id (categorical) x shuffle_level (continuous 0-4).
Reports eta-squared for image, shuffle, and interaction in all / ani / inani scopes.
'''

#%% Directory and imports

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from tqdm import tqdm
import OS_Tools as ot

datapath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\ANOVA'
brain_areas = ['ML', 'MSB', 'AL', 'ASB','ALO']

WINDOW_S = 0.17
N_REPEAT = 5
N_SHUF = 5
N_IMG_OBJ = 40
N_COL = N_REPEAT * N_SHUF * N_IMG_OBJ   # 1000 columns per cell
ANI_IMG = np.arange(20)
INANI_IMG = np.arange(20, 40)

# Layout: 5 repeats x (5 shuffle levels x 40 images)
IMG_ID = np.tile(np.arange(N_IMG_OBJ), N_REPEAT * N_SHUF)
SHUFFLE_LEVEL = np.tile(np.repeat(np.arange(N_SHUF), N_IMG_OBJ), N_REPEAT)
assert len(IMG_ID) == len(SHUFFLE_LEVEL) == N_COL

FIT_SCOPES = [
    ('all', None),
    ('ani', True),
    ('inani', False),
]

RESULT_COLS = [
    'eta2_img', 'eta2_shuffle', 'eta2_inter', 'eta2_resid',
    'F_img', 'F_shuffle', 'F_inter',
    'p_img', 'p_shuffle', 'p_inter',
    'n_obs', 'n_valid_obs',
]


#%% Functions

def scope_mask(scope_flag):
    if scope_flag is None:
        return np.ones(len(IMG_ID), dtype=bool)
    if scope_flag:
        return np.isin(IMG_ID, ANI_IMG)
    return np.isin(IMG_ID, INANI_IMG)


def load_area(area):
    area_dir = ot.Join(datapath, area)
    rsp_hz = np.load(ot.Join(area_dir, 'avr_rsp.npy')) / WINDOW_S
    if rsp_hz.shape[1] != N_COL:
        raise ValueError(f'{area}: expected {N_COL} columns, got {rsp_hz.shape[1]}')
    info = pd.read_csv(ot.Join(area_dir, 'cell_site_info.csv'))
    return rsp_hz, info


def anova_one_cell(y, img, shuf):
    """Type II ANOVA; shuffle as linear, img as categorical. Returns dict or NaNs."""
    y = np.asarray(y, dtype=float)
    img = np.asarray(img)
    shuf = np.asarray(shuf, dtype=float)
    n_obs = len(y)
    valid = np.isfinite(y) & np.isfinite(shuf)
    n_valid = int(valid.sum())

    out = {c: np.nan for c in RESULT_COLS}
    out['n_obs'] = n_obs
    out['n_valid_obs'] = n_valid

    if n_valid < 10 or np.nanstd(y[valid]) < 1e-12:
        return out

    df = pd.DataFrame({
        'response': y[valid],
        'img_id': img[valid],
        'shuffle_level': shuf[valid],
    })
    try:
        model = ols('response ~ C(img_id) * shuffle_level', data=df).fit()
        anova_res = anova_lm(model, typ=2)
    except Exception:
        return out

    total_ss = anova_res['sum_sq'].sum()
    if not np.isfinite(total_ss) or total_ss <= 0:
        return out

    img_row = anova_res.loc['C(img_id)']
    shuf_row = anova_res.loc['shuffle_level']
    inter_row = anova_res.loc['C(img_id):shuffle_level']
    resid_row = anova_res.loc['Residual']

    out['eta2_img'] = img_row['sum_sq'] / total_ss
    out['eta2_shuffle'] = shuf_row['sum_sq'] / total_ss
    out['eta2_inter'] = inter_row['sum_sq'] / total_ss
    out['eta2_resid'] = resid_row['sum_sq'] / total_ss
    out['F_img'] = img_row['F']
    out['F_shuffle'] = shuf_row['F']
    out['F_inter'] = inter_row['F']
    out['p_img'] = img_row['PR(>F)']
    out['p_shuffle'] = shuf_row['PR(>F)']
    out['p_inter'] = inter_row['PR(>F)']
    return out


def run_scope(rsp_hz, scope_name, scope_flag):
    mask = scope_mask(scope_flag)
    img = IMG_ID[mask]
    shuf = SHUFFLE_LEVEL[mask]
    n_cell = rsp_hz.shape[0]
    rows = []
    for i in tqdm(range(n_cell), desc=scope_name, leave=False):
        row = anova_one_cell(rsp_hz[i, mask], img, shuf)
        row['cell_idx'] = i
        row['image_scope'] = scope_name
        rows.append(row)
    return pd.DataFrame(rows)


def attach_meta(df, info):
    meta_cols = ['global_idx', 'site_name', 'dprime_face', 'dprime_body']
    present = [c for c in meta_cols if c in info.columns]
    if not present:
        return df
    meta = info[present].copy()
    meta['cell_idx'] = meta.index
    return df.merge(meta, on='cell_idx', how='left')


#%% Main loop

ot.Mkdir(savepath)
all_dfs = []

for area in brain_areas:
    rsp_hz, info = load_area(area)
    area_dfs = []

    for scope_name, scope_flag in FIT_SCOPES:
        df_scope = run_scope(rsp_hz, scope_name, scope_flag)
        df_scope.insert(0, 'area', area)
        df_scope = attach_meta(df_scope, info)

        out_dir = ot.Join(savepath, area)
        ot.Mkdir(out_dir)
        df_scope.to_csv(ot.Join(out_dir, f'anova_2fac_{scope_name}.csv'), index=False)
        area_dfs.append(df_scope)
        n_ok = df_scope['eta2_img'].notna().sum()
        print(f'{area} {scope_name}: {len(df_scope)} rows, {n_ok} fitted')

    all_dfs.extend(area_dfs)

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv(ot.Join(savepath, 'anova_2fac_all_cells.csv'), index=False)
print(f'saved all: {len(df_all)} rows -> {ot.Join(savepath, "anova_2fac_all_cells.csv")}')

for scope_name, _ in FIT_SCOPES:
    scope_df = df_all[df_all['image_scope'] == scope_name]
    scope_df.to_csv(ot.Join(savepath, f'anova_2fac_{scope_name}.csv'), index=False)
    print(f'saved {scope_name}: {len(scope_df)} rows')

#%%

import matplotlib.pyplot as plt
import seaborn as sns

PLOT_SCOPE = 'all'   # 'all' | 'ani' | 'inani'
FIG_DIR = ot.Join(savepath, 'figures')
ot.Mkdir(FIG_DIR)

sub = df_all[df_all['image_scope'] == PLOT_SCOPE].copy()
sub['r2'] = 1 - sub['eta2_resid']
sub = sub[np.isfinite(sub['r2'])]

# --- Fig 1: model R2 density by area ---
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(
    data=sub, x='r2', hue='area', stat='density',
    element='step', fill=True, common_norm=False, ax=ax,
)
ax.set_xlabel('Model R²')
ax.set_ylabel('Density')
ax.set_title(f'2-way ANOVA model R²  ({PLOT_SCOPE})')
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(ot.Join(FIG_DIR, f'anova_r2_hist_{PLOT_SCOPE}.png'), dpi=150, bbox_inches='tight')
plt.show()

#%% --- Fig 2: eta² boxplot by effect and area ---
PLOT_SCOPE = 'ani'   # 'all' | 'ani' | 'inani'
effect_map = {
    'eta2_img': 'Image ID',
    'eta2_shuffle': 'Shuffle level',
    'eta2_inter': 'Interaction',
}
long = sub.melt(
    id_vars='area',
    value_vars=list(effect_map),
    var_name='effect',
    value_name='eta2',
)
long['effect'] = long['effect'].map(effect_map)
long = long[np.isfinite(long['eta2'])]

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(
    data=long, x='effect', y='eta2', hue='area',
    order=list(effect_map.values()), ax=ax,showfliers=False
)
ax.set_xlabel('')
ax.set_ylabel('Explained variance (η²)')
ax.set_title(f'ANOVA explained variance  ({PLOT_SCOPE})')
ax.set_ylim(0, None)
fig.tight_layout()
fig.savefig(ot.Join(FIG_DIR, f'anova_eta2_box_{PLOT_SCOPE}.png'), dpi=150, bbox_inches='tight')
plt.show()

#%%

ALPHA = 0.05
EFFECTS = [
    ('p_img', 'Image ID'),
    ('p_shuffle', 'Shuffle level'),
    ('p_inter', 'Interaction'),
]

prop_rows = []
for (area, scope), g in df_all.groupby(['area', 'image_scope'], sort=False):
    for pcol, label in EFFECTS:
        p = g[pcol].dropna()
        prop_rows.append({
            'area': area,
            'image_scope': scope,
            'effect': label,
            'prop_sig': (p < ALPHA).mean() if len(p) else np.nan,
            'n_sig': int((p < ALPHA).sum()),
            'n_cell': len(p),
        })

prop_df = pd.DataFrame(prop_rows)
prop_df.to_csv(ot.Join(savepath, 'anova_sig_prop.csv'), index=False)
print(prop_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, scope in zip(axes, ['all', 'ani', 'inani']):
    sns.barplot(
        data=prop_df[prop_df['image_scope'] == scope],
        x='area', y='prop_sig', hue='effect', ax=ax,
        order=brain_areas,
        hue_order=[e[1] for e in EFFECTS],
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel('')
    ax.set_ylabel('Fraction significant (p < 0.05)')
    ax.set_title(scope)
    ax.legend(title='', fontsize=8, loc='upper right')
fig.suptitle('Significant neurons by area and image scope', y=1.02)
fig.tight_layout()
fig.savefig(ot.Join(FIG_DIR, 'anova_sig_prop_bar.png'), dpi=150, bbox_inches='tight')
plt.show()
#%%

from scipy.stats import mannwhitneyu

# --- plot knobs (None = auto); bracket sizes are per-metric y-axis units ---
# bracket_base_y: y of the lowest bracket bottom (inani + lower compare pair);
#                 higher brackets stack upward from here.
PLOT_KW = {
    'cohen_f_shuffle': {
        'ylim': (-0.05, 1),
        'bracket_base_y': 0.76,#最下面一条线的位置
        'bracket_h': 0.015,# barcket竖条纹的长度
        'bracket_row_step': 0.015,
        'bracket_scope_step': 0.045,# 每个bracket之间的距离，主要调这个
        'bracket_top_pad': 0.025,
    },
    'slope_hz': {
        'ylim': (-1.5, 2),
        'bracket_base_y': 1.15,
        'bracket_h': 0.06,
        'bracket_row_step': 0.08,
        'bracket_scope_step': 0.12,
        'bracket_top_pad': 0.06,
    },
}

def bracket_cfg(metric):
    m = PLOT_KW.get(metric, {})
    span = (m['ylim'][1] - m['ylim'][0]) if m.get('ylim') else 1.0
    return {k: m.get(k, span * f) for k, f in [
        ('bracket_h', 0.012), ('bracket_row_step', 0.018),
        ('bracket_scope_step', 0.028), ('bracket_top_pad', 0.015),
    ]} | {'bracket_base_y': m.get('bracket_base_y')}

def bracket_bottom_y(j, k, bk, y_top):
    """y of bracket bottom; anchor at bracket_base_y or fall back to top padding."""
    bh = bk['bracket_h']
    row_step, scope_step = bk['bracket_row_step'], bk['bracket_scope_step']
    base_y = bk.get('bracket_base_y')
    if base_y is not None:
        j_bot, k_bot = len(SCOPE_ORDER) - 1, len(COMPARE_PAIRS) - 1
        return (base_y
                + (k_bot - k) * (bh + row_step)
                + (j_bot - j) * (bh + scope_step))
    y_scope = y_top - bk['bracket_top_pad'] - bh - j * (bh + scope_step)
    return y_scope - k * (bh + row_step)

def _patch_xcenter(patch):
    """Box x center for Rectangle (old mpl) or PathPatch (new mpl/seaborn)."""
    if hasattr(patch, 'get_x') and hasattr(patch, 'get_width'):
        return patch.get_x() + patch.get_width() / 2
    xs = patch.get_path().vertices[:, 0]
    return float((xs.min() + xs.max()) / 2)

def box_xcenters(ax):
    """Read dodged box x centers; fallback to manual offsets if needed."""
    dodge = 0.8 / len(SCOPE_ORDER)
    pos = {}
    for j, scope in enumerate(SCOPE_ORDER):
        off = (j - (len(SCOPE_ORDER) - 1) / 2) * dodge
        for k, area in enumerate(AREA_ORDER):
            pos[(scope, area)] = k + off
    for j, container in enumerate(ax.containers):
        if j >= len(SCOPE_ORDER):
            break
        scope = SCOPE_ORDER[j]
        patches = getattr(container, 'patches', None) or getattr(container, 'boxes', None)
        if not patches:
            continue
        for k, patch in enumerate(patches):
            if k < len(AREA_ORDER):
                pos[(scope, AREA_ORDER[k])] = _patch_xcenter(patch)
    return pos

SCOPE_IMGS = {'all': np.arange(N_IMG_OBJ), 'ani': ANI_IMG, 'inani': INANI_IMG}
SCOPE_ORDER = ['all', 'ani', 'inani']
COMPARE_PAIRS = [('MSB', 'ASB'), ('ML', 'AL')]
AREA_ORDER = ['AL', 'ML', 'ASB', 'MSB']

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

def add_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c='k')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=9)

def cohen_f_shuffle(eta2_shuffle, eta2_resid):
    """Cohen's f for shuffle level from partial η² = SS_shuffle / (SS_shuffle + SS_resid)."""
    eta2_shuffle = np.asarray(eta2_shuffle, dtype=float)
    eta2_resid = np.asarray(eta2_resid, dtype=float)
    denom = eta2_shuffle + eta2_resid
    peta2 = np.full_like(eta2_shuffle, np.nan)
    valid = np.isfinite(eta2_shuffle) & np.isfinite(eta2_resid) & (denom > 0)
    peta2[valid] = eta2_shuffle[valid] / denom[valid]
    f = np.full_like(peta2, np.nan)
    ok = np.isfinite(peta2) & (peta2 < 1)
    f[ok] = np.sqrt(peta2[ok] / (1 - peta2[ok]))
    return f


def cell_slope_hz(r4, img_ids):
    lvl = r4[:, :, :, img_ids].mean(axis=(1, 3))
    slope = np.full(r4.shape[0], np.nan)
    ok = np.isfinite(lvl).all(axis=1)
    slope[ok] = np.polyfit(np.arange(N_SHUF), lvl[ok].T, 1, cov=False)[0]
    return slope


metric_df = df_all[['area', 'cell_idx', 'image_scope', 'eta2_shuffle', 'eta2_resid']].copy()
metric_df['cohen_f_shuffle'] = cohen_f_shuffle(
    metric_df['eta2_shuffle'].values, metric_df['eta2_resid'].values,
)

slope_rows = []
for area in brain_areas:
    rsp_hz = np.load(ot.Join(datapath, area, 'avr_rsp.npy')) / WINDOW_S
    r4 = rsp_hz.reshape(-1, N_REPEAT, N_SHUF, N_IMG_OBJ)
    for scope_name, img_ids in SCOPE_IMGS.items():
        slope_rows.append(pd.DataFrame({
            'area': area, 'cell_idx': np.arange(r4.shape[0]),
            'image_scope': scope_name, 'slope_hz': cell_slope_hz(r4, img_ids),
        }))
metric_df = metric_df.merge(
    pd.concat(slope_rows, ignore_index=True),
    on=['area', 'cell_idx', 'image_scope'],
    how='left',
)

def plot_metric_box(metric, ylabel, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=metric_df, x='area', y=metric, hue='image_scope',
        order=AREA_ORDER, hue_order=SCOPE_ORDER, ax=ax, showfliers=False,
    )
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(title='', fontsize=8)

    ylim = PLOT_KW.get(metric, {}).get('ylim')
    if ylim is not None:
        ax.set_ylim(ylim)
    y_top = ax.get_ylim()[1]
    bk = bracket_cfg(metric)
    bh = bk['bracket_h']
    xpos = box_xcenters(ax)
    sub_m = metric_df.dropna(subset=[metric])
    for j, scope in enumerate(SCOPE_ORDER):
        for k, (a1, a2) in enumerate(COMPARE_PAIRS):
            v1 = sub_m.loc[(sub_m['area'] == a1) & (sub_m['image_scope'] == scope), metric]
            v2 = sub_m.loc[(sub_m['area'] == a2) & (sub_m['image_scope'] == scope), metric]
            if len(v1) and len(v2):
                _, p = mannwhitneyu(v1, v2, alternative='two-sided')
                y = bracket_bottom_y(j, k, bk, y_top)
                add_bracket(ax, xpos[(scope, a1)], xpos[(scope, a2)], y, bh, p_to_star(p))

    fig.tight_layout()
    fig.savefig(ot.Join(FIG_DIR, f'anova_{fname}_box.png'), dpi=150, bbox_inches='tight')
    plt.show()

plot_metric_box('cohen_f_shuffle', "Cohen's f (shuffle level, ANOVA)", 'cohen_f_shuffle')
plot_metric_box('slope_hz', 'Marginal slope (Hz / shuffle level)', 'slope_hz')
