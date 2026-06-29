#%% imports and parameters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import OS_Tools as ot

datapath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\decay_index'
FIG_DIR = ot.Join(savepath, 'figures', 'Decay_Index_Stats')

WINDOW_S = 0.17
N_REPEAT = 5
N_SHUF = 5
N_IMG = 40
COLOR_ANI = '#c0392b'
COLOR_INANI = '#2980b9'
COLOR_ALL = '#2c3e50'
BRAIN_AREAS = ['ML', 'MSB', 'AL', 'ASB']
AREA_ORDER = ['ASB', 'MSB', 'AL', 'ML']
AREA_COLORS = dict(zip(AREA_ORDER, ['#c0392b', '#2980b9', '#27ae60', '#8e44ad']))





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


def plot_violin_box(ax, data, positions, colors, violin_w=0.65, box_w=0.18, whis=BOX_WHIS):
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


# --- demo inputs ---
DEMO_AREA = 'ASB'
DEMO_CELL = 902
DEMO_IMG = [ 8,12, 31]   # int or list; 0-based index: 0=img1


#%% demo: decay curve for one image + raw/beta pattern across 40 images

def load_cell_data(area):
    """Load per-image beta table and raw shuffle-level responses for one area."""
    df = pd.read_csv(ot.Join(savepath, area, 'decay_beta_by_image.csv'))
    rsp = np.load(ot.Join(datapath, area, 'avr_rsp.npy'))
    rsp_hz = rsp / WINDOW_S
    r4 = rsp_hz.reshape(-1, N_REPEAT, N_SHUF, N_IMG)
    return df, r4


def _as_img_list(img_id):
    if np.isscalar(img_id):
        return [int(img_id)]
    return [int(i) for i in img_id]


def plot_neuron_demo(area, cell_idx, img_id):
    """img_id: int or list of 0-based indices; displayed as img 1..40."""
    img_ids = _as_img_list(img_id)
    df, r4 = load_cell_data(area)
    sub = df[df['cell_idx'] == cell_idx]

    levels = np.arange(N_SHUF, dtype=float)
    raw_all = sub['raw_rsp'].to_numpy()
    beta_all = sub['decay_beta'].to_numpy()
    colors = [COLOR_ANI if i < 20 else COLOR_INANI for i in range(N_IMG)]
    img_ticks = np.arange(1, N_IMG + 1)
    curve_colors = plt.cm.tab10(np.linspace(0, 1, max(len(img_ids), 1)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1) decay curves for selected images
    ax = axes[0]
    for i, iid in enumerate(img_ids):
        img_no = iid + 1
        c = curve_colors[i]
        raw_img = r4[cell_idx, :, 0, iid].mean()
        change = raw_img - r4[cell_idx, :, :, iid].mean(0)
        row = sub[sub['img_id'] == iid].iloc[0]
        beta, r2 = row['decay_beta'], row['r2']

        ax.scatter(levels, change, s=40, color=c, zorder=3)
        ax.plot(levels, beta * raw_img * levels, '-', color=c, lw=2,
                label=f'img {img_no}: beta={beta:.3f}, R2={r2:.2f}')

    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xticks(levels)
    ax.set_xlabel('Shuffle level')
    ax.set_ylabel('Firing rate change (Hz)  raw - response')
    img_nos = [i + 1 for i in img_ids]
    ax.set_title(f'{area} cell {cell_idx}  img {img_nos}')
    ax.legend(fontsize=7)

    # 2) raw response across 40 images
    ax = axes[1]
    ax.bar(img_ticks, raw_all, color=colors, width=0.8, edgecolor='none')
    for i, iid in enumerate(img_ids):
        ax.axvline(iid + 1, color=curve_colors[i], ls='--', lw=1.2)
    ax.set_xlim(0.5, N_IMG + 0.5)
    ax.set_xlabel('Image')
    ax.set_ylabel('Raw firing rate (Hz)')
    ax.set_title('Raw response pattern')

    # 3) beta across 40 images
    ax = axes[2]
    ax.bar(img_ticks, beta_all, color=colors, width=0.8, edgecolor='none')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    for i, iid in enumerate(img_ids):
        ax.axvline(iid + 1, color=curve_colors[i], ls='--', lw=1.2)
    ax.set_xlim(0.5, N_IMG + 0.5)
    ax.set_xlabel('Image')
    ax.set_ylabel('Beta  (decay per shuffle level)')
    ax.set_title('Beta pattern')

    fig.suptitle(f'{area}  cell {cell_idx}  |  red=ani  blue=inani', fontsize=11)
    fig.tight_layout()

    ot.Mkdir(FIG_DIR)
    img_tag = '_'.join(str(i + 1) for i in img_ids)
    out = ot.Join(FIG_DIR, f'demo_{area}_cell{cell_idx}_img{img_tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'saved: {out}')


plot_neuron_demo(DEMO_AREA, DEMO_CELL, DEMO_IMG)

#%% heatmap: beta + raw response, rows=neurons, cols=40 images

HEAT_AREA = 'ML'

df = pd.read_csv(ot.Join(savepath, HEAT_AREA, 'decay_beta_by_image.csv'))
beta_wide = df.pivot(index='cell_idx', columns='img_id', values='decay_beta')
raw_wide = df.pivot(index='cell_idx', columns='img_id', values='raw_rsp')

order = beta_wide.mean(axis=1).argsort()
beta_mat = beta_wide.to_numpy()[order]
raw_mat = raw_wide.to_numpy()[order]

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

vlim = np.nanpercentile(np.abs(beta_mat), 98)
im0 = axes[0].imshow(beta_mat, aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
axes[0].axvline(19.5, color='k', lw=1)
axes[0].set_ylabel('Neuron  (sorted by mean beta)')
axes[0].set_title('Beta')
fig.colorbar(im0, ax=axes[0], label='Beta', shrink=0.6)

vmax = np.nanpercentile(raw_mat, 98)
im1 = axes[1].imshow(raw_mat, aspect='auto', cmap='hot', vmin=5, vmax=vmax-3)
axes[1].axvline(19.5, color='w', lw=1)
axes[1].set_title('Raw response (Hz)')
fig.colorbar(im1, ax=axes[1], label='Hz', shrink=0.6)

for ax in axes:
    ax.set_xticks([])

fig.suptitle(f'{HEAT_AREA}  n={beta_mat.shape[0]} cells', fontsize=11)
fig.tight_layout()

ot.Mkdir(FIG_DIR)
out = ot.Join(FIG_DIR, f'heatmap_beta_raw_{HEAT_AREA}.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print(f'saved: {out}')
#%% within-neuron raw-beta correlation by area (violin + box)

# raw-beta corr violin plot (last section)
CORR_YLIM = (-0.3,1.2)              # e.g. (-0.5, 0.9); None = auto from data
CORR_YLIM_PAD = 0.05          # padding when CORR_YLIM is None
CORR_COMPARE_PAIRS = [('ASB', 'MSB'), ('AL', 'ML')]
CORR_BRACKET_Y = [1.1, 1.1] # base y (data coords) for each pair in CORR_COMPARE_PAIRS
CORR_BRACKET_H = 0.025        # vertical arm height of each bracket
CORR_BRACKET_FS = 9
BOX_WHIS = (10,90)   # box whisker percentiles; e.g. (10, 90) or 1.5 for IQR

df_all = pd.read_csv(ot.Join(savepath, 'decay_beta_by_image_all.csv'))
rows = []
for (area, cell_idx), sub in df_all.groupby(['area', 'cell_idx'], sort=False):
    valid = sub['raw_valid'] & np.isfinite(sub['raw_rsp']) & np.isfinite(sub['decay_beta'])
    v = sub.loc[valid]
    if len(v) < 3:
        continue
    raw = v['raw_rsp'].to_numpy(dtype=float)
    beta = v['decay_beta'].to_numpy(dtype=float)
    if np.nanstd(raw) == 0 or np.nanstd(beta) == 0:
        continue
    r, _ = stats.pearsonr(raw, beta)
    rows.append({'area': area, 'cell_idx': cell_idx, 'r': r})
corr_df = pd.DataFrame(rows)

data = [corr_df.loc[corr_df['area'] == a, 'r'].to_numpy() for a in AREA_ORDER]
colors = [AREA_COLORS[a] for a in AREA_ORDER]
positions = np.arange(len(AREA_ORDER))
area_pos = {a: i for i, a in enumerate(AREA_ORDER)}

fig, ax = plt.subplots(figsize=(6, 5))
plot_violin_box(ax, data, positions, colors, whis=BOX_WHIS)

ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(AREA_ORDER)
ax.set_ylabel('Within-neuron corr(raw, beta)')
ax.set_title('Raw response vs beta binding by area')

if CORR_YLIM is not None:
    ax.set_ylim(CORR_YLIM)
else:
    all_r = corr_df['r'].to_numpy(dtype=float)
    ymin = float(np.nanmin(all_r)) - CORR_YLIM_PAD
    ymax = float(np.nanmax(all_r)) + CORR_YLIM_PAD
    if CORR_BRACKET_Y:
        ymax = max(ymax, max(CORR_BRACKET_Y) + CORR_BRACKET_H + 0.02)
    ax.set_ylim(ymin, ymax)

for (a1, a2), y_br in zip(CORR_COMPARE_PAIRS, CORR_BRACKET_Y):
    v1 = corr_df.loc[corr_df['area'] == a1, 'r'].dropna().to_numpy(dtype=float)
    v2 = corr_df.loc[corr_df['area'] == a2, 'r'].dropna().to_numpy(dtype=float)
    if len(v1) < 3 or len(v2) < 3:
        print(f'{a1} vs {a2}: insufficient data (n={len(v1)}, {len(v2)})')
        continue
    u, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
    print(f'{a1} vs {a2}: Mann-Whitney U={u:.1f}, p={p:.4g}, n=({len(v1)}, {len(v2)})')
    add_sig_bracket(
        ax, area_pos[a1], area_pos[a2], y_br, CORR_BRACKET_H,
        p_to_star(p), fs=CORR_BRACKET_FS,
    )

fig.tight_layout()

ot.Mkdir(FIG_DIR)
out = ot.Join(FIG_DIR, 'raw_beta_corr_by_area.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print(f'saved: {out}')


#%% beta distribution by area x image scope (ani / inani / all)

BETA_SCOPES = [
    ('ani', True, COLOR_ANI, 'Ani'),
    ('inani', False, COLOR_INANI, 'Inani'),
    ('all', None, COLOR_ALL, 'All'),
]
BETA_SCOPE_OFFSETS = [-0.28, 0.0, 0.28]
BETA_VIOLIN_W = 0.25
BETA_BOX_W = 0.08
BETA_BOX_WHIS = (10, 90)
BETA_YLIM = (-0.2,0.4)          # e.g. (-0.05, 0.35); None = auto from data
BETA_YLIM_PAD = 0.02
BETA_ANI_COMPARE_PAIRS = [('ASB', 'MSB'), ('AL', 'ML')]
BETA_ANI_BRACKET_Y = [0.35, 0.35]   # base y for each pair (ani bars only)
BETA_ANI_BRACKET_H = 0.02
BETA_ANI_BRACKET_FS = 9

df_beta = pd.read_csv(ot.Join(savepath, 'decay_beta_by_image_all.csv'))
valid_beta = df_beta['raw_valid'] & np.isfinite(df_beta['decay_beta'])
area_pos = {a: i for i, a in enumerate(AREA_ORDER)}
ani_offset = BETA_SCOPE_OFFSETS[0]


def get_scope_beta(area, scope_flag):
    sub = df_beta[df_beta['area'] == area]
    mask = valid_beta.loc[sub.index]
    if scope_flag is not None:
        mask = mask & sub['is_ani'].eq(scope_flag)
    return sub.loc[mask, 'decay_beta'].to_numpy(dtype=float)

data, positions, colors = [], [], []
for i, area in enumerate(AREA_ORDER):
    sub = df_beta[df_beta['area'] == area]
    for (_, scope_flag, scope_color, _), offset in zip(BETA_SCOPES, BETA_SCOPE_OFFSETS):
        mask = valid_beta.loc[sub.index]
        if scope_flag is not None:
            mask = mask & sub['is_ani'].eq(scope_flag)
        vals = sub.loc[mask, 'decay_beta'].to_numpy(dtype=float)
        data.append(vals)
        positions.append(i + offset)
        colors.append(scope_color)

fig, ax = plt.subplots(figsize=(7, 5))
plot_violin_box(
    ax, data, positions, colors,
    violin_w=BETA_VIOLIN_W, box_w=BETA_BOX_W, whis=BETA_BOX_WHIS,
)

ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xticks(np.arange(len(AREA_ORDER)))
ax.set_xticklabels(AREA_ORDER)
ax.set_ylabel('Beta  (decay per shuffle level)')
ax.set_title('Beta distribution by area')

if BETA_YLIM is not None:
    ax.set_ylim(BETA_YLIM)
else:
    all_beta = df_beta.loc[valid_beta, 'decay_beta'].to_numpy(dtype=float)
    pad = BETA_YLIM_PAD
    ax.set_ylim(float(np.nanmin(all_beta)) - pad, float(np.nanmax(all_beta)) + pad)

for (a1, a2), y_br in zip(BETA_ANI_COMPARE_PAIRS, BETA_ANI_BRACKET_Y):
    v1 = get_scope_beta(a1, True)
    v2 = get_scope_beta(a2, True)
    if len(v1) < 3 or len(v2) < 3:
        print(f'ani {a1} vs {a2}: insufficient data (n={len(v1)}, {len(v2)})')
        continue
    u, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
    print(f'ani {a1} vs {a2}: Mann-Whitney U={u:.1f}, p={p:.4g}, n=({len(v1)}, {len(v2)})')
    add_sig_bracket(
        ax,
        area_pos[a1] + ani_offset,
        area_pos[a2] + ani_offset,
        y_br,
        BETA_ANI_BRACKET_H,
        p_to_star(p),
        fs=BETA_ANI_BRACKET_FS,
    )

from matplotlib.patches import Patch
ax.legend(
    handles=[Patch(facecolor=c, edgecolor=c, alpha=0.6, label=lbl)
             for _, _, c, lbl in BETA_SCOPES],
    loc='upper right', frameon=False, fontsize=9,
)

for area in AREA_ORDER:
    for scope_name, scope_flag, _, scope_label in BETA_SCOPES:
        sub = df_beta[df_beta['area'] == area]
        mask = valid_beta.loc[sub.index]
        if scope_flag is not None:
            mask = mask & sub['is_ani'].eq(scope_flag)
        vals = sub.loc[mask, 'decay_beta'].to_numpy(dtype=float)
        print(f'{area} {scope_label}: n={len(vals)}, median={np.nanmedian(vals):.4f}')

fig.tight_layout()

ot.Mkdir(FIG_DIR)
out = ot.Join(FIG_DIR, 'beta_distribution_by_area_scope.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print(f'saved: {out}')
