'''

For each plot, average its response to ani and inani, then fit a 

'''

#%%


import seaborn as sns
import OS_Tools as ot
from PIL import Image
import numpy as np
from Matrix_Tools import Corr_Matrix
import pandas as pd
from scipy.stats import pearsonr, linregress, wilcoxon, ttest_ind
from itertools import permutations
import random
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


savepath = r'E:\#Preprocessed_Data\260305_Report_Data\Firing_Rate_Slope'

FR_All = pd.read_parquet(ot.Join(savepath,'FR_Raw.parquet'))

# 1) Mean Raw_FR through different Img_Index → one row per (Brain_Area, Cell, Shuffle, Ani)
# 2) Normalize so each cell's Shuffle==0 at Ani and inani is 1 (divide by image-averaged Raw_FR at Shuffle==0)
group_cols = ['Brain_Area', 'Cell', 'Shuffle', 'Ani']
FR_Img_AVR = FR_All.groupby(group_cols, as_index=False)['Raw_FR'].mean()
# Baseline per (Brain_Area, Cell, Ani) = value at Shuffle==0
shuffle0 = FR_Img_AVR.loc[FR_Img_AVR['Shuffle'] == 0, ['Brain_Area', 'Cell', 'Ani', 'Raw_FR']].rename(columns={'Raw_FR': '_baseline'})
FR_Img_AVR = FR_Img_AVR.merge(shuffle0, on=['Brain_Area', 'Cell', 'Ani'], how='left')
FR_Img_AVR['Normed_FR'] = np.where(
    (FR_Img_AVR['_baseline'] > 0) & FR_Img_AVR['_baseline'].notna(),
    FR_Img_AVR['Raw_FR'] / FR_Img_AVR['_baseline'],
    np.nan
)
FR_Img_AVR = FR_Img_AVR.drop(columns=['_baseline'])
# Sort so each (Brain_Area, Cell, Ani) has 5 consecutive rows (Shuffle 0..4)
FR_Img_AVR = FR_Img_AVR.sort_values(['Brain_Area', 'Cell', 'Ani', 'Shuffle']).reset_index(drop=True)


#%%
n_fits = FR_Img_AVR.groupby(['Brain_Area', 'Cell', 'Ani'], sort=False).ngroups
Fit_Info_Real = pd.DataFrame(index=range(n_fits), columns=['Brain_Area','Cell','Ani','Slope','Intercept','R2','P'])
Fit_Info_Bootstrap = pd.DataFrame(index=range(n_fits), columns=['Brain_Area','Cell','Ani','Slope','Intercept','R2','P'])

def _derangements(n):
    """All derangements of range(n) (permutations with no fixed point)."""
    for p in permutations(range(n)):
        if all(p[i] != i for i in range(n)):
            yield list(p)

def _linear_fit_r2(x, y):
    """Fit y = slope*x + intercept; return slope, intercept, R2, pvalue (for slope)."""
    res = linregress(x, y)
    r2 = res.rvalue ** 2 if res.rvalue is not None else np.nan
    return res.slope, res.intercept, r2, res.pvalue

# All 44 derangements of [0,1,2,3,4]
DERANGEMENTS_5 = np.array(list(_derangements(5)))
n_shuffle = 5

row = 0
for (ba, cell, ani), grp in tqdm(FR_Img_AVR.groupby(['Brain_Area', 'Cell', 'Ani'], sort=False)):
    grp = grp.sort_values('Shuffle')
    x = grp['Shuffle'].to_numpy(dtype=float)
    y = grp['Normed_FR'].to_numpy()
    if len(x) != n_shuffle:
        Fit_Info_Real.loc[row, ['Brain_Area', 'Cell', 'Ani']] = ba, cell, ani
        Fit_Info_Bootstrap.loc[row, ['Brain_Area', 'Cell', 'Ani']] = ba, cell, ani
        Fit_Info_Real.loc[row, ['Slope', 'Intercept', 'R2', 'P']] = np.nan, np.nan, np.nan, np.nan
        Fit_Info_Bootstrap.loc[row, ['Slope', 'Intercept', 'R2', 'P']] = np.nan, np.nan, np.nan, np.nan
        row += 1
        continue
    slope_real, intercept_real, r2_real, p_real = _linear_fit_r2(x, y)
    d = DERANGEMENTS_5[random.randrange(len(DERANGEMENTS_5))]
    y_boot = y[d]
    slope_boot, intercept_boot, r2_boot, p_boot = _linear_fit_r2(x, y_boot)
    Fit_Info_Real.loc[row, ['Brain_Area', 'Cell', 'Ani', 'Slope', 'Intercept', 'R2', 'P']] = (
        ba, cell, ani, slope_real, intercept_real, r2_real, p_real)
    Fit_Info_Bootstrap.loc[row, ['Brain_Area', 'Cell', 'Ani', 'Slope', 'Intercept', 'R2', 'P']] = (
        ba, cell, ani, slope_boot, intercept_boot, r2_boot, p_boot)
    row += 1

#%%
Fit_Info_Real.to_parquet(ot.Join(savepath,'FR_Slope_Fit_AVR_Real.parquet'))
Fit_Info_Bootstrap.to_parquet(ot.Join(savepath,'FR_Slope_Fit_AVR_Bootstrap.parquet'))

#%%
############## Plot parts ###############


def _lighten_color(color, amount=0.5):
    """Lighten the given color by mixing it with white."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _p_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def _add_sig_bracket(ax, x1, x2, y, text, line_height_frac=0.02):
    """Draw a short horizontal marker and put text above it."""
    if text == '':
        return
    y_min, y_max = ax.get_ylim()
    h = abs(y_max - y_min) * line_height_frac
    x_center = (x1 + x2) / 2.0
    is_inverted = y_min > y_max
    marker_half_width = max(0.04, 0.18 * abs(x2 - x1))

    if is_inverted:
        # In inverted axis, smaller y is visually higher.
        y_text = y - 0.20 * h
        va = 'top'
    else:
        y_text = y + 0.20 * h
        va = 'bottom'

    ax.plot(
        [x_center - marker_half_width, x_center + marker_half_width],
        [y, y],
        color='k',
        linewidth=1,
        clip_on=True,
    )
    ax.text(x_center, y_text, text, ha='center', va=va, clip_on=True, fontsize=8)


def _safe_sig_y(y, y_top, y_bottom, line_height_frac=0.01):
    """Clamp annotation y so line+text remains inside axis range."""
    span = y_bottom - y_top
    pad = max(0.012, (line_height_frac * 1.8 + 0.01) * span)
    return float(np.clip(y, y_top + pad, y_bottom - pad))


def plot_slope_summary(
    Fit_Info_Real,
    Fit_Info_Bootstrap,
    in_group_sig_y=-0.25,
    between_area_sig_y=-0.35,
    between_area_sig_step=0.015,
    y_top = -0.4,
    y_bottom = 0.21
):
    """Two subplots: Ani==1 (animated) and Ani==0 (inanimated) with real vs bootstrap slopes."""
    # Order of brain areas along x-axis
    area_order = ['MSB', 'ML', 'ASB', 'AL']

    # Define base colors per area (for real) and lighter variants (for bootstrap)
    area_colors = {
        'MSB': '#0072B2',  # blue
        'ML': '#D55E00',   # orange
        'ASB': '#009E73',  # green
        'AL': '#CC79A7',   # pink
    }
    area_colors_light = {k: _lighten_color(v, amount=0.6) for k, v in area_colors.items()}
    # Fixed display range requested by user.

    y_span = y_bottom - y_top

    # Pair real and bootstrap slopes per cell/area/Ani for within-area Wilcoxon
    paired = Fit_Info_Real.merge(
        Fit_Info_Bootstrap,
        on=['Brain_Area', 'Cell', 'Ani'],
        suffixes=('_real', '_boot'),
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, dpi=240, figsize=(10, 5), sharey=False)

    neighbor_pairs = [('MSB', 'ML'), ('ML', 'ASB'), ('ASB', 'AL')]

    for col, ani_val in enumerate([1, 0]):
        ax = axes[col]

        real_ani = Fit_Info_Real[Fit_Info_Real['Ani'] == ani_val].copy()
        boot_ani = Fit_Info_Bootstrap[Fit_Info_Bootstrap['Ani'] == ani_val].copy()
        paired_ani = paired[paired['Ani'] == ani_val].copy()

        mean_x = []
        mean_y = []

        for i, area in enumerate(area_order):
            s_real = real_ani.loc[real_ani['Brain_Area'] == area, 'Slope'].dropna()
            s_boot = boot_ani.loc[boot_ani['Brain_Area'] == area, 'Slope'].dropna()

            if len(s_real) == 0 and len(s_boot) == 0:
                continue

            pos_real = i - 0.15
            pos_boot = i + 0.15

            bp = ax.boxplot(
                [s_real, s_boot],
                positions=[pos_real, pos_boot],
                widths=0.25,
                patch_artist=True,
                showfliers=False,
            )

            # Color boxes: real = strong color, bootstrap = lighter color
            bp['boxes'][0].set_facecolor(area_colors[area])
            bp['boxes'][1].set_facecolor(area_colors_light[area])

            for whisker in bp['whiskers']:
                whisker.set_color('black')
            for cap in bp['caps']:
                cap.set_color('black')
            for median in bp['medians']:
                median.set_color('black')

            # Within-area Wilcoxon (paired real vs bootstrap per cell, one-sided: real < bootstrap)
            paired_area = paired_ani[paired_ani['Brain_Area'] == area]

            # Convert to float arrays and keep only finite pairs
            s_r = paired_area['Slope_real'].to_numpy(dtype=float)
            s_b = paired_area['Slope_boot'].to_numpy(dtype=float)
            mask = np.isfinite(s_r) & np.isfinite(s_b)
            s_r = s_r[mask]
            s_b = s_b[mask]

            if len(s_r) >= 5 and len(s_r) == len(s_b):
                diff = s_r - s_b
                # need at least one non-zero difference for Wilcoxon
                if np.any(diff != 0):
                    stat, p = wilcoxon(s_r, s_b, alternative='less')
                    stars = _p_to_stars(p)
                    if stars:
                        # Manual in-group annotation height (user-adjustable parameter).
                        y = _safe_sig_y(
                            in_group_sig_y,
                            y_top,
                            y_bottom,
                            line_height_frac=0.01,
                        )
                        _add_sig_bracket(ax, pos_real, pos_boot, y, stars, line_height_frac=0.01)

            # Collect mean of real slopes for mean-line plot
            if len(s_real) > 0:
                mean_x.append(pos_real)
                mean_y.append(float(np.nanmean(s_real)))

        # Draw line connecting mean real slopes across brain areas (behind boxplots)
        if mean_x and mean_y:
            ax.plot(
                mean_x,
                mean_y,
                color='black',
                linestyle='-',
                linewidth=1,
                marker='o',
                markersize=3,
                zorder=1,
            )

        # Between-area tests on real slopes, neighboring areas (Welch t-test)
        for j, (a1, a2) in enumerate(neighbor_pairs):
            # Between-area comparison uses real slopes only; coerce to float and drop non-finite
            s1 = real_ani.loc[real_ani['Brain_Area'] == a1, 'Slope'].to_numpy(dtype=float)
            s2 = real_ani.loc[real_ani['Brain_Area'] == a2, 'Slope'].to_numpy(dtype=float)
            s1 = s1[np.isfinite(s1)]
            s2 = s2[np.isfinite(s2)]
            if len(s1) >= 5 and len(s2) >= 5:
                # Welch t-test (independent samples, unequal variance)
                stat, p = ttest_ind(s1, s2, equal_var=False, nan_policy='omit')
                stars = _p_to_stars(p) if np.isfinite(p) else ''
                label = f"T={stat:.2f}" + (f" {stars}" if stars else "")
                # Bracket over the real-slope box positions for the two areas
                idx1 = area_order.index(a1)
                idx2 = area_order.index(a2)
                x1 = idx1 - 0.15
                x2 = idx2 - 0.15
                # Manual between-area annotation height (user-adjustable parameter).
                y = between_area_sig_y + j * between_area_sig_step
                y = _safe_sig_y(
                    y,
                    y_top,
                    y_bottom,
                    line_height_frac=0.01,
                )
                _add_sig_bracket(ax, x1, x2, y, label, line_height_frac=0.01)

        ax.set_xticks(range(len(area_order)))
        ax.set_xticklabels(area_order)
        ax.set_xlabel('Brain Area')
        if col == 0:
            ax.set_ylabel('Slope')

        ax.set_title('Animate' if ani_val == 1 else 'Inanimate')
        # Zero line behind boxplots and mean line
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=0)
        # Fixed range with negative slopes at the top.
        ax.set_ylim(y_bottom, y_top)

    plt.tight_layout()
    plt.show()

# Call the summary plotting function
plot_slope_summary(
    Fit_Info_Real,
    Fit_Info_Bootstrap,
    in_group_sig_y=-0.25,
    between_area_sig_y=-0.35,
)







#%% ##### Plot an example cell's response.
cell = 'MSB_394'
ba = 'MSB'
FR_Cell = FR_Img_AVR.loc[
    (FR_Img_AVR['Cell'] == cell) &
    (FR_Img_AVR['Brain_Area'] == ba)
].copy()  # contains both Ani==True and Ani==False

fig, ax = plt.subplots(figsize=(6, 4), dpi=240, sharey=True)

colors = {1: 'tab:orange', 0: 'tab:blue'}
labels = {1: 'Animated', 0: 'Inanimated'}

for ani in [1, 0]:
    sub = FR_Cell.loc[FR_Cell['Ani'] == ani].sort_values('Shuffle')
    x = sub['Shuffle'].to_numpy(dtype=float)
    y = sub['Raw_FR'].to_numpy(dtype=float)

    # Plot the scatter for data points
    ax.scatter(x, y, color=colors[ani], marker='o', label=f"{labels[ani]} scatter")

    # Also plot the fit line
    if len(x) >= 2 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
        slope, intercept, _, _ = _linear_fit_r2(x, y)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color=colors[ani], linestyle='--', label=f"{labels[ani]} fit")

ax.set_title(f'Example Cell: {cell}')
ax.set_xlabel('Plot Order')
ax.set_ylabel('Raw_Firing_Rate')
ax.legend()
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(['Raw','C4','C3','C2','C1'])
ax.set_ylim([0,1.5])
plt.tight_layout()
plt.show()


#%% ###########################################################################
#%% Per-image slope fits (no image-averaging first)

# For this section we work directly on FR_All at the single-image level.
single_group_cols = ['Brain_Area', 'Cell', 'Ani', 'Img_Index']

# Baseline per (Brain_Area, Cell, Ani, Img_Index) at Shuffle==0
shuffle0_single = (
    FR_All.loc[FR_All['Shuffle'] == 0, single_group_cols + ['Raw_FR']]
    .rename(columns={'Raw_FR': '_baseline'})
)

FR_Single = FR_All.merge(shuffle0_single, on=single_group_cols, how='left')
FR_Single['Normed_FR'] = np.where(
    (FR_Single['_baseline'] > 0) & FR_Single['_baseline'].notna(),
    FR_Single['Raw_FR'] / FR_Single['_baseline'],
    np.nan,
)
FR_Single = FR_Single.drop(columns=['_baseline'])

# Sort so each (Brain_Area, Cell, Ani, Img_Index) has 5 consecutive rows (Shuffle 0..4)
FR_Single = FR_Single.sort_values(
    ['Brain_Area', 'Cell', 'Ani', 'Img_Index', 'Shuffle']
).reset_index(drop=True)

# Fit slopes per (Brain_Area, Cell, Ani, Img_Index)
fit_group_cols_single = ['Brain_Area', 'Cell', 'Ani', 'Img_Index']
n_fits_single = FR_Single.groupby(fit_group_cols_single, sort=False).ngroups

Fit_Info_Real_Single = pd.DataFrame(
    index=range(n_fits_single),
    columns=['Brain_Area', 'Cell', 'Ani', 'Img_Index', 'Slope', 'Intercept', 'R2', 'P'],
)
Fit_Info_Bootstrap_Single = pd.DataFrame(
    index=range(n_fits_single),
    columns=['Brain_Area', 'Cell', 'Ani', 'Img_Index', 'Slope', 'Intercept', 'R2', 'P'],
)

row = 0
for (ba, cell, ani, img_idx), grp in tqdm(
    FR_Single.groupby(fit_group_cols_single, sort=False)
):
    grp = grp.sort_values('Shuffle')
    x = grp['Shuffle'].to_numpy(dtype=float)
    y = grp['Raw_FR'].to_numpy()
    if len(x) != n_shuffle:
        Fit_Info_Real_Single.loc[row, ['Brain_Area', 'Cell', 'Ani', 'Img_Index']] = (
            ba,
            cell,
            ani,
            img_idx,
        )
        Fit_Info_Bootstrap_Single.loc[
            row, ['Brain_Area', 'Cell', 'Ani', 'Img_Index']
        ] = (ba, cell, ani, img_idx)
        Fit_Info_Real_Single.loc[row, ['Slope', 'Intercept', 'R2', 'P']] = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
        Fit_Info_Bootstrap_Single.loc[row, ['Slope', 'Intercept', 'R2', 'P']] = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
        row += 1
        continue

    slope_real, intercept_real, r2_real, p_real = _linear_fit_r2(x, y)
    d = DERANGEMENTS_5[random.randrange(len(DERANGEMENTS_5))]
    y_boot = y[d]
    slope_boot, intercept_boot, r2_boot, p_boot = _linear_fit_r2(x, y_boot)

    Fit_Info_Real_Single.loc[
        row, ['Brain_Area', 'Cell', 'Ani', 'Img_Index', 'Slope', 'Intercept', 'R2', 'P']
    ] = (ba, cell, ani, img_idx, slope_real, intercept_real, r2_real, p_real)
    Fit_Info_Bootstrap_Single.loc[
        row, ['Brain_Area', 'Cell', 'Ani', 'Img_Index', 'Slope', 'Intercept', 'R2', 'P']
    ] = (ba, cell, ani, img_idx, slope_boot, intercept_boot, r2_boot, p_boot)
    row += 1

# Save per-image fit information
Fit_Info_Real_Single.to_parquet(
    ot.Join(savepath, 'FR_Slope_Fit_SingleImg_Real.parquet')
)
Fit_Info_Bootstrap_Single.to_parquet(
    ot.Join(savepath, 'FR_Slope_Fit_SingleImg_Bootstrap.parquet')
)

#%% Plot summary for per-image slopes using the same plotting function
plot_slope_summary(Fit_Info_Real_Single, Fit_Info_Bootstrap_Single,y_top = -0.45,y_bottom = 0.3,in_group_sig_y=-0.32,between_area_sig_y=-0.4)




