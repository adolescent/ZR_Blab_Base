'''
Stats of number of significant slop<0 cells in each brain area.

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

FR_Avr_Real = pd.read_parquet(ot.Join(savepath, 'FR_Slope_Fit_AVR_Real.parquet'))
FR_SingleImg_Real = pd.read_parquet(ot.Join(savepath, 'FR_Slope_Fit_SingleImg_Real.parquet'))

#%%
# For FR_Avr_Real, compute in each brain area (and separately for ani / inani)
# the proportion of cells whose slope < 0 and whose p-value is significant.
# Result is a DataFrame with columns:
# ['Area', 'Ani', 'Prop_p<0.01', 'Prop_p<0.05']

rows = []
for (area, ani), df_grp in FR_Avr_Real.groupby(['Brain_Area', 'Ani']):
    # Keep only finite slope/p pairs
    valid = df_grp[['Slope', 'P']].replace([np.inf, -np.inf], np.nan).dropna()
    n_total = len(valid)
    if n_total == 0:
        prop_p001 = np.nan
        prop_p005 = np.nan
    else:
        neg = valid['Slope'] < 0
        p = valid['P']
        prop_p001 = ((neg) & (p < 0.01)).sum() / n_total
        prop_p005 = ((neg) & (p < 0.05)).sum() / n_total

    rows.append(
        {
            'Area': area,
            'Ani': ani,
            'Prop_p<0.01': prop_p001,
            'Prop_p<0.05': prop_p005,
        }
    )

neg_slope_sig_summary = pd.DataFrame(rows, columns=['Area', 'Ani', 'Prop_p<0.01', 'Prop_p<0.05'])

# At this point neg_slope_sig_summary is the requested pandas DataFrame.

#%%
fig, axes = plt.subplots(1, 2, figsize=(8, 5), dpi=240, sharey=True)
# Ensure plot_df is filtered, integer Ani, and labeled
plot_df = neg_slope_sig_summary.copy()
ani_map = {1: 'Animate', 0: 'Inanimate'}
plot_df['Ani'] = plot_df['Ani'].fillna(-1).astype(int)
plot_df = plot_df[plot_df['Ani'].isin([0, 1])].copy()
plot_df['Ani_Label'] = plot_df['Ani'].map(ani_map)

# Fixed x label order
area_order = ['MSB', 'ML', 'ASB', 'AL']
plot_df = plot_df[plot_df['Area'].isin(area_order)].copy()
plot_df['Area'] = pd.Categorical(plot_df['Area'], categories=area_order, ordered=True)

# Colors: Animate = orange, Inanimate = blue; hue_order keeps Animate left, Inanimate right
color_map = {'Animate': 'tab:orange', 'Inanimate': 'tab:blue'}
hue_order = ['Animate', 'Inanimate']  # drawn in order: Animate left, Inanimate right

props = [
    ('Prop_p<0.05', 'Significant negative slope (p<0.05)'),
    ('Prop_p<0.01', 'Significant negative slope (p<0.01)')
]

for ax, (prop_col, title) in zip(axes, props):
    # Draw bar plot explicitly with manual colors
    for i, area in enumerate(area_order):
        for j, ani_label in enumerate(hue_order):
            value = plot_df.loc[
                (plot_df['Area'] == area) & (plot_df['Ani_Label'] == ani_label), prop_col
            ]
            if not value.empty:
                ax.bar(
                    i + j * 0.3 - 0.15,  # j=0 Animate left, j=1 Inanimate right
                    value.values[0],
                    width=0.28,
                    color=color_map[ani_label],
                    label=ani_label if (i == 0) else None,  # Only show label in legend once
                    edgecolor='black',
                    linewidth=0.7,
                )
    ax.set_xticks(range(len(area_order)))
    ax.set_xticklabels(area_order)
    ax.set_xlabel('Brain Area')
    ax.set_title(title)
axes[0].set_ylabel('Proportion')
# Custom legend on left subplot only
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes[1].legend(by_label.values(), by_label.keys(), title='Stimulus Type')
plt.tight_layout()
plt.show()

#%%
######################### Single Image #################################
'''
This part is to calculate significant slope<0 cells for single image.
For each cell, count the number of images with significant negative slope (slope<0 and p sig).
Count Animate and Inanimate separately.
Output: neg_slope_single_sig_summary with columns [Area, Cell, Ani, N_sig_p001, N_sig_p005].
'''

rows_single = []
for (area, cell, ani), df_grp in FR_SingleImg_Real.groupby(['Brain_Area', 'Cell', 'Ani']):
    valid = df_grp[['Slope', 'P']].replace([np.inf, -np.inf], np.nan).dropna()
    neg = valid['Slope'] < 0
    p = valid['P']
    n_sig_p001 = ((neg) & (p < 0.01)).sum()
    n_sig_p005 = ((neg) & (p < 0.05)).sum()
    rows_single.append({
        'Area': area,
        'Cell': cell,
        'Ani': ani,
        'N_sig_p001': int(n_sig_p001),
        'N_sig_p005': int(n_sig_p005),
    })

neg_slope_single_sig_summary = pd.DataFrame(
    rows_single,
    columns=['Area', 'Cell', 'Ani', 'N_sig_p001', 'N_sig_p005'],
)

#%%
# Plot 1, stats of n_sig images for each cell.
fig, axes = plt.subplots(1, 2, figsize=(8, 5), dpi=240, sharey=True)
# Ensure plot_df is filtered, integer Ani, and labeled
plot_df = neg_slope_single_sig_summary.copy()
ani_map = {1: 'Animate', 0: 'Inanimate'}
plot_df['Ani'] = plot_df['Ani'].fillna(-1).astype(int)
plot_df = plot_df[plot_df['Ani'].isin([0, 1])].copy()
plot_df['Ani_Label'] = plot_df['Ani'].map(ani_map)

# Fixed x label order
area_order = ['MSB', 'ML', 'ASB', 'AL']
plot_df = plot_df[plot_df['Area'].isin(area_order)].copy()
plot_df['Area'] = pd.Categorical(plot_df['Area'], categories=area_order, ordered=True)

for idx, (p_val, title) in enumerate(zip(['N_sig_p001', 'N_sig_p005'], ['p < 0.01', 'p < 0.05'])):
    ax = axes[idx]
    sns.boxplot(
        data=plot_df, 
        x='Area', 
        y=p_val, 
        hue='Ani_Label', 
        showcaps=True, 
        showmeans=False, 
        meanline=False, 
        ax=ax,
        showfliers=False,whis=[5, 95]
    )
    ax.set_title(f"# Images with sig neg slope ({title})")
    ax.set_ylabel('Image Count')
    ax.set_xlabel('Area')
    if idx == 1:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Stimulus Type')
    else:
        ax.get_legend().remove()

plt.tight_layout()
plt.show()

#%%
# Plot 2, Count prop of cells with at least 1 image with sig neg slope.
neg_slope_single_sig_summary['Signum>1'] = neg_slope_single_sig_summary['N_sig_p005'] > 1
neg_slope_single_sig_summary['Signum>2'] = neg_slope_single_sig_summary['N_sig_p005'] > 2

# Proportion of cells with Signum>1 / Signum>2 per (Area, Ani), ani-inani in each brain area
prop_single_rows = []
for (area, ani), grp in neg_slope_single_sig_summary.groupby(['Area', 'Ani']):
    n_cells = len(grp)
    prop_single_rows.append({
        'Area': area,
        'Ani': ani,
        'Prop_Signum_gt1': grp['Signum>1'].mean(),
        'Prop_Signum_gt2': grp['Signum>2'].mean(),
    })
prop_single_summary = pd.DataFrame(prop_single_rows)

# Plot 2 subplots: left = Signum>1 proportion, right = Signum>2 proportion
fig2, axes2 = plt.subplots(1, 2, figsize=(8, 5), dpi=240, sharey=True)
plot_df2 = prop_single_summary.copy()
ani_map = {1: 'Animate', 0: 'Inanimate'}
plot_df2['Ani'] = plot_df2['Ani'].fillna(-1).astype(int)
plot_df2 = plot_df2[plot_df2['Ani'].isin([0, 1])].copy()
plot_df2['Ani_Label'] = plot_df2['Ani'].map(ani_map)

area_order = ['MSB', 'ML', 'ASB', 'AL']
plot_df2 = plot_df2[plot_df2['Area'].isin(area_order)].copy()
color_map = {'Animate': 'tab:orange', 'Inanimate': 'tab:blue'}
hue_order = ['Animate', 'Inanimate']

for ax, (prop_col, title) in zip(axes2, [('Prop_Signum_gt1', 'Sig Img>1'), ('Prop_Signum_gt2', 'Sig Img>2')]):
    for i, area in enumerate(area_order):
        for j, ani_label in enumerate(hue_order):
            value = plot_df2.loc[
                (plot_df2['Area'] == area) & (plot_df2['Ani_Label'] == ani_label), prop_col
            ]
            if not value.empty:
                ax.bar(
                    i + j * 0.3 - 0.15,
                    value.values[0],
                    width=0.28,
                    color=color_map[ani_label],
                    label=ani_label if (i == 0) else None,
                    edgecolor='black',
                    linewidth=0.7,
                )
    ax.set_xticks(range(len(area_order)))
    ax.set_xticklabels(area_order)
    ax.set_xlabel('Brain Area')
    ax.set_title(title)
axes2[0].set_ylabel('Proportion')
handles2, labels2 = axes2[0].get_legend_handles_labels()
by_label2 = dict(zip(labels2, handles2))
axes2[1].legend(by_label2.values(), by_label2.keys(), title='Stimulus Type')
plt.tight_layout()
plt.show()

#%%


