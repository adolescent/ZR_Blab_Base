

'''
Stat and demo of 2-way anova results.
For each stimsets and for every brain areas.


'''
#%%

import os
import matplotlib
import pandas as pd
import OS_Tools as ot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from matplotlib.patches import PathPatch


datafolder = r'E:\#Preprocessed_Data\260305_Report_Data\Site_ANOVAs'
namelists = ot.Get_File_Name(datafolder, '.parquet')


#%%
# Concat all parquet files and add brain area / Ani-or-not columns

all_tables = []

for p in namelists:
    df = pd.read_parquet(p)

    fname = os.path.basename(p)
    parts = fname.split('_')

    # Expected patterns:
    #   Ani_AL_ANOVA.parquet
    #   Inani_AL_ANOVA.parquet
    #   All_AL_ANOVA.parquet
    prefix = parts[0] if len(parts) > 0 else ''
    brain_area = parts[1] if len(parts) > 1 else 'Unknown'

    is_ani = prefix  # True for Ani_*, False otherwise

    df['Brain_Area'] = brain_area
    df['Is_Ani'] = is_ani

    all_tables.append(df)

ANOVA_ALL = pd.concat(all_tables, ignore_index=True)


#%%

# Define custom color palettes: All in gray, Ani in orange, Inani in blue
# Same Brain_Area appears in All/Ani/Inani, so palette must be keyed by (Is_Ani, Brain_Area)
ANOVA_ALL['Hue_Key'] = ANOVA_ALL['Is_Ani'] + '_' + ANOVA_ALL['Brain_Area']

ani_areas   = sorted(ANOVA_ALL.loc[ANOVA_ALL['Is_Ani'].str.lower() == 'ani', 'Brain_Area'].unique())
inani_areas = sorted(ANOVA_ALL.loc[ANOVA_ALL['Is_Ani'].str.lower() == 'inani', 'Brain_Area'].unique())
all_areas   = sorted(ANOVA_ALL.loc[ANOVA_ALL['Is_Ani'].str.lower() == 'all', 'Brain_Area'].unique())

# Set explicit order for x ("Is_Ani") and hue ("Hue_Key" = Is_Ani_Brain_Area)
is_ani_order = []
if len(all_areas):   is_ani_order.append("All")
if len(ani_areas):   is_ani_order.append("Ani")
if len(inani_areas): is_ani_order.append("Inani")

# Hue order: All_AL, All_AM, ..., Ani_AL, ..., Inani_AL, ...
hue_order = [f'All_{a}' for a in all_areas] + [f'Ani_{a}' for a in ani_areas] + [f'Inani_{a}' for a in inani_areas]

palette = {}
if len(all_areas):
    if len(all_areas) == 1:
        all_colors = [(0.5, 0.5, 0.5)]
    else:
        all_colors = sns.color_palette("Greys", n_colors=max(3, len(all_areas)+2))[2:2+len(all_areas)]
    for area, c in zip(all_areas, all_colors):
        palette[f'All_{area}'] = c
if len(ani_areas):
    if len(ani_areas) == 1:
        ani_colors = [(1.0, 0.5, 0.1)]
    else:
        ani_colors = sns.color_palette("Oranges", n_colors=max(3, len(ani_areas)+2))[2:2+len(ani_areas)]
    for area, c in zip(ani_areas, ani_colors):
        palette[f'Ani_{area}'] = c
if len(inani_areas):
    if len(inani_areas) == 1:
        inani_colors = [(0.25, 0.4, 1.0)]
    else:
        inani_colors = sns.color_palette("Blues", n_colors=max(3, len(inani_areas)+2))[2:2+len(inani_areas)]
    for area, c in zip(inani_areas, inani_colors):
        palette[f'Inani_{area}'] = c

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), dpi=240)

def grouped_boxplot(
    ax,
    df: pd.DataFrame,
    metric: str,
    *,
    x_order,
    hue_order,
    palette,
    group_width: float = 1.5,   # total width of the whole hue group at one x tick
    spacing: float = 1.0,       # x-tick distance (seaborn categories are 0,1,2,...)
    whis=(5, 95),
    showfliers: bool = False,
):
    """
    Draw centered grouped boxplots (per x tick) with explicit positions, so groups like
    All_* / Ani_* / Inani_* are centered under their category even when hue levels differ by x.
    """
    x_centers = {x: i * spacing for i, x in enumerate(x_order)}

    for x in x_order:
        df_x = df[df['Is_Ani'] == x]
        if df_x.empty:
            continue

        hues_here = [h for h in hue_order if h in set(df_x['Hue_Key'].unique())]
        n_h = len(hues_here)
        if n_h == 0:
            continue

        box_w = group_width / n_h

        for hi, h in enumerate(hues_here):
            pos = x_centers[x] + (hi - (n_h - 1) / 2) * box_w
            vals = df_x.loc[df_x['Hue_Key'] == h, metric].dropna().values
            if len(vals) == 0:
                continue

            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_w * 0.95,
                whis=whis,
                showfliers=showfliers,
                patch_artist=True,
                manage_ticks=False,
            )

            face = palette.get(h, (0.7, 0.7, 0.7))
            for patch in bp['boxes']:
                patch.set_facecolor(face)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.0)
            for med in bp['medians']:
                med.set_color('black')
                med.set_linewidth(1.2)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(1.0)
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(1.0)

    ax.set_xticks([x_centers[x] for x in x_order])
    ax.set_xticklabels(x_order)


grouped_boxplot(ax[0, 0], ANOVA_ALL, 'Explained_VAR_ALL',      x_order=is_ani_order, hue_order=hue_order, palette=palette, group_width=0.8, spacing=1.0, whis=(5, 95))
grouped_boxplot(ax[0, 1], ANOVA_ALL, 'Explained_VAR_Shuffle',  x_order=is_ani_order, hue_order=hue_order, palette=palette, group_width=0.8, spacing=1.0, whis=(5, 95))
grouped_boxplot(ax[1, 0], ANOVA_ALL, 'Explained_VAR_Img',      x_order=is_ani_order, hue_order=hue_order, palette=palette, group_width=0.8, spacing=1.0, whis=(5, 95))
grouped_boxplot(ax[1, 1], ANOVA_ALL, 'Explained_VAR_Interact', x_order=is_ani_order, hue_order=hue_order, palette=palette, group_width=0.8, spacing=1.0, whis=(5, 95))

ax[0, 0].set_title('VAR Explained All')
ax[0, 1].set_title('VAR Explaiend Constrain Level')
ax[1, 0].set_title('VAR Explaiend Img')
ax[1, 1].set_title('VAR Explained Interact')

# Set specific y-limits for each subplot as per instruction
ax[0, 0].set_ylim(0, 0.9)    # Fig 1: Explained_VAR_ALL
ax[1, 0].set_ylim(0, 0.8)    # Fig 3: Explained_VAR_Img
ax[0, 1].set_ylim(0, 0.15)   # Fig 2: Explained_VAR_Shuffle
ax[1, 1].set_ylim(0, 0.2)    # Fig 4: Explained_VAR_Interact

# Keep all groups centered and give some padding at edges
for a in ax.ravel():
    a.set_xlim(-0.6, len(is_ani_order) - 0.4)

# Build a single legend (custom boxplots don't auto-create one)
from matplotlib.patches import Patch
handles = [Patch(facecolor=palette[h], edgecolor='black', label=h) for h in hue_order if h in palette]
labels = [h for h in hue_order if h in palette]
leg = ax[0, 1].legend(
    handles, labels,
    title="",
    fontsize=7,
    title_fontsize=10,
    loc='upper left',
    bbox_to_anchor=(0.78, 0.90),
    borderaxespad=0.5,
    frameon=True,
    framealpha=0.8,
    fancybox=True,
    handlelength=1,
    handleheight=0.7,
    borderpad=0.5,
    columnspacing=0.8,
    labelspacing=0.3,
    ncol=1
)

# Shrink legend patch sizes (matplotlib version differences)
legend_handles = getattr(leg, 'legend_handles', None)
if legend_handles is None:
    legend_handles = getattr(leg, 'legendHandles', None)
if legend_handles is not None:
    for handle in legend_handles:
        if hasattr(handle, "set_width"):
            handle.set_width(5)
        if hasattr(handle, "set_height"):
            handle.set_height(2)
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(0)


# 
# Significance annotations: pooled Ani vs Inani (ignore brain area) for each metric panel

def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def add_sig_bar_ani_vs_inani_pooled(ax, metric_col: str):
    # Import locally so this works regardless of cell execution order (notebook usage)
    from scipy.stats import mannwhitneyu

    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    h = 0.012 * yr
    y = y1 - 0.06 * yr

    # Get x-centers for the category ticks (Ani and Inani)
    tick_pos = dict(zip([t.get_text() for t in ax.get_xticklabels()], ax.get_xticks()))
    if ('Ani' not in tick_pos) or ('Inani' not in tick_pos):
        return

    x1 = float(tick_pos['Ani'])
    x2 = float(tick_pos['Inani'])

    a = ANOVA_ALL[ANOVA_ALL['Is_Ani'] == 'Ani'][metric_col].dropna()
    b = ANOVA_ALL[ANOVA_ALL['Is_Ani'] == 'Inani'][metric_col].dropna()
    if len(a) == 0 or len(b) == 0:
        return

    p = mannwhitneyu(a, b, alternative='two-sided').pvalue
    stars = _p_to_stars(float(p))
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='k', lw=1.0, clip_on=False)
    # ax.text((x1 + x2) / 2.0, y + h, f'{stars}\np={p:.2e}',
    #         ha='center', va='bottom', fontsize=9, color='k', clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h, f'{stars}',
    ha='center', va='bottom', fontsize=9, color='k', clip_on=False)


add_sig_bar_ani_vs_inani_pooled(ax[0, 0], 'Explained_VAR_ALL')
add_sig_bar_ani_vs_inani_pooled(ax[0, 1], 'Explained_VAR_Shuffle')
add_sig_bar_ani_vs_inani_pooled(ax[1, 0], 'Explained_VAR_Img')
add_sig_bar_ani_vs_inani_pooled(ax[1, 1], 'Explained_VAR_Interact')


# 
######### stat significant constrain alone cells.
# %%
# Proportion of cells with significant p_Shuffle per (Brain_Area, Is_Ani)

def _p_props(s: pd.Series) -> pd.Series:
    # Returns a Series with two values, which are then expanded into separate columns automatically
    return pd.Series([(s < 0.05).mean(), (s < 0.01).mean()], index=['p_005_prop', 'p_001_prop'])

p_shuffle_summary = (
    ANOVA_ALL
    .groupby(['Brain_Area', 'Is_Ani'])['p_Shuffle']
    .apply(_p_props)
    .reset_index()
)

# At this point columns will be: Brain_Area, Is_Ani, p_005_prop, p_001_prop
# No need to select only these columns - they are present and in order

print(p_shuffle_summary)

# %%
# Bar plots of p_Shuffle proportions, reusing the same colors as above

# Recreate Hue_Key so we can reuse hue_order & palette
p_shuffle_summary['Hue_Key'] = p_shuffle_summary['Is_Ani'] + '_' + p_shuffle_summary['Brain_Area']

# Clean up/remap output from .apply result so column names are what downstream expects.

# When you do .groupby(...).apply(func).reset_index(), if func returns a pd.Series with a custom index,
# pandas puts the index as columns: ['Brain_Area','Is_Ani','level_2','p_Shuffle']
# To get 'p_005_prop' and 'p_001_prop' in columns, pivot this df:
if 'level_2' in p_shuffle_summary.columns and 'p_Shuffle' in p_shuffle_summary.columns:
    # Wide to long reshape
    p_shuffle_summary_pivot = p_shuffle_summary.pivot(index=['Brain_Area','Is_Ani','Hue_Key'], 
                                                      columns='level_2', 
                                                      values='p_Shuffle').reset_index()
    # Now columns are: ['Brain_Area', 'Is_Ani', 'Hue_Key', 0, 1]
    # We set correct names
    # Copy for safety, and name columns
    p_shuffle_summary_pivot = p_shuffle_summary_pivot.copy()
    p_shuffle_summary_pivot = p_shuffle_summary_pivot.rename(columns={0:'p_005_prop', 1:'p_001_prop'})
else:
    # Already in expected format
    p_shuffle_summary_pivot = p_shuffle_summary.copy()

print("Columns in p_shuffle_summary_pivot:", p_shuffle_summary_pivot.columns)

y_cols = ['p_005_prop', 'p_001_prop']
for ycol in y_cols:
    if ycol not in p_shuffle_summary_pivot.columns:
        raise ValueError(f"Column {ycol} not found in p_shuffle_summary_pivot. Found: {p_shuffle_summary_pivot.columns}")

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4), dpi=240)

def grouped_barplot(
    ax,
    df: pd.DataFrame,
    y: str,
    *,
    x_order,
    hue_order,
    palette,
    group_width: float = 2.0,   # total width of the whole hue group
    spacing: float = 3.0,       # distance between x-tick centers (must be > group_width)
):
    x_centers = {x: i * spacing for i, x in enumerate(x_order)}

    # Center bars within each x tick using only hues that exist at that x.
    for x in x_order:
        df_x = df[df['Is_Ani'] == x]
        if df_x.empty:
            continue

        hues_here = [h for h in hue_order if h in set(df_x['Hue_Key'].unique())]
        n_h = max(1, len(hues_here))
        bar_w = group_width / n_h

        for hi, h in enumerate(hues_here):
            offset = (hi - (n_h - 1) / 2) * bar_w
            val = df_x.loc[df_x['Hue_Key'] == h, y].mean()

            ax.bar(
                x_centers[x] + offset,
                val,
                width=bar_w * 0.95,
                color=palette.get(h, None),
                label=h,
            )

    ax.set_xticks([x_centers[x] for x in x_order])
    ax.set_xticklabels(x_order)
    ax.set_xlim(min(x_centers.values()) - spacing * 0.5, max(x_centers.values()) + spacing * 0.5)


grouped_barplot(ax2[0], p_shuffle_summary_pivot, y_cols[0], x_order=is_ani_order, hue_order=hue_order, palette=palette, group_width=2.0, spacing=3.0)
grouped_barplot(ax2[1], p_shuffle_summary_pivot, y_cols[1], x_order=is_ani_order, hue_order=hue_order, palette=palette, group_width=2.0, spacing=3.0)

ax2[0].set_title('p_Shuffle < 0.05 (proportion)')
ax2[1].set_title('p_Shuffle < 0.01 (proportion)')

for a in ax2:
    a.set_ylim(0, 1)

# One shared legend on the first bar subplot
for a in ax2:
    lg = a.get_legend()
    if lg is not None:
        lg.remove()

handles2, labels2 = ax2[1].get_legend_handles_labels()
leg2 = ax2[1].legend(
    handles2, labels2,
    title="",
    fontsize=7,
    loc='upper right',
    frameon=True,
    framealpha=0.8,
)


# %%

