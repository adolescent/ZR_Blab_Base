'''
Visualize parque result in A3.
'''

#%%
import Common_Functions.OS_Tools as ot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% ###########################Plot Corrs Here#################################


wp = r'E:\#Preprocessed_Data\260402_TC_Analysis'
brain_areas = ['MSB','ASB','AL','ML']
used_ares = brain_areas[1]
t_ms0 = 0

all_img_corr = pd.read_parquet(
    ot.Join(wp, f'{used_ares}_shuffle_timecorr_psth_pdf_win10_step5.parquet')
)

plot_shuffle_a = 0
plot_shuffle_b = 4
plot_cell_group = 'c2'
vmax=0.1
avg_img_indices = np.arange(1, 21)

def plot_shuffle_timecorr_heatmap_mean_imgs(
    df,
    img_indices,
    shuffle_a,
    shuffle_b,
    cell_group='c0',
    t_ms_offset=0,
    tick_ms_step=60,
    plot_t_start_ms=None,
    plot_t_end_ms=None,
    figsize=(8, 8),
    save_path=None,
    vmax=1.0,
):
    """Average corr_mean over Img_Index, then same heatmap as plot_shuffle_timecorr_heatmap."""
    img_indices = np.unique(np.asarray(img_indices, dtype=int).ravel())
    sub = df.loc[
        df['Img_Index'].isin(img_indices)
        & (df['shuffle_level_A'] == int(shuffle_a))
        & (df['shuffle_level_B'] == int(shuffle_b))
        & (df['cell_group'] == str(cell_group))
    ]
    toff = int(t_ms_offset)
    if plot_t_start_ms is not None:
        rel_start = int(plot_t_start_ms) - toff
        sub = sub.loc[(sub['win_start_A'] >= rel_start) & (sub['win_start_B'] >= rel_start)]
    if plot_t_end_ms is not None:
        rel_end = int(plot_t_end_ms) - toff
        sub = sub.loc[(sub['win_start_A'] <= rel_end) & (sub['win_start_B'] <= rel_end)]
    if sub.empty:
        raise ValueError(
            'No rows for img_indices / shuffle A,B / cell_group / plot time range.'
        )
    n_imgs_used = int(sub['Img_Index'].nunique())
    agg = sub.groupby(['win_start_A', 'win_start_B'], as_index=False).agg(
        corr_mean=('corr_mean', 'mean'),
        n_cells=('n_cells', 'first'),
        win_len=('win_len', 'first'),
        brain_area=('brain_area', 'first'),
    )
    pivot = agg.pivot_table(
        index='win_start_A',
        columns='win_start_B',
        values='corr_mean',
        aggfunc='first',
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    n_cells = int(agg['n_cells'].iloc[0])
    wl = int(agg['win_len'].iloc[0])
    brain_area = str(agg['brain_area'].iloc[0])
    pivot_plot = pivot.iloc[::-1, :]
    rel_a = pivot_plot.index.to_numpy(dtype=int)
    rel_b = pivot.columns.to_numpy(dtype=int)
    step = int(tick_ms_step)

    def _sparse_ms_labels(rel_starts):
        return [
            f'{toff + int(r)}' if int(r) % step == 0 else ''
            for r in rel_starts
        ]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_plot.to_numpy(dtype=float),
        ax=ax,
        cmap='coolwarm',
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        xticklabels=_sparse_ms_labels(rel_b),
        yticklabels=_sparse_ms_labels(rel_a),
        cbar_kws={
            'label': 'mean Pearson r (mean over imgs)',
            'shrink': 0.55,
            'aspect': 28,
        },
    )
    row_labels = pivot_plot.index.to_numpy(dtype=int)
    col_labels = pivot_plot.columns.to_numpy(dtype=int)
    row_pos = {int(v): i for i, v in enumerate(row_labels)}
    col_pos = {int(v): j for j, v in enumerate(col_labels)}
    common_starts = sorted(set(row_pos.keys()) & set(col_pos.keys()))
    if common_starts:
        xs = [col_pos[v] + 0.5 for v in common_starts]
        ys = [row_pos[v] + 0.5 for v in common_starts]
        ax.plot(
            xs,
            ys,
            color=(0.92, 0.92, 0.92),
            linewidth=1.4,
            linestyle='-',
            zorder=5,
            clip_on=True,
        )
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'Time window B start (ms)\n(win_len={wl}; shuffle_level_B={int(shuffle_b)})')
    ax.set_ylabel(f'Time window A start (ms)\n(win_len={wl}; shuffle_level_A={int(shuffle_a)})')
    img_rng = f'{int(img_indices.min())}-{int(img_indices.max())}'
    ax.set_title(
        f'{brain_area} | {cell_group} (n={n_cells} cells)\n'
        f'Mean over Img_Index ∈ [{img_rng}] (n_imgs={n_imgs_used}) | '
        f'shuffle A={int(shuffle_a)} vs B={int(shuffle_b)}'
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax



fig_avg, ax_avg = plot_shuffle_timecorr_heatmap_mean_imgs(
    all_img_corr,
    avg_img_indices,
    plot_shuffle_a,
    plot_shuffle_b,
    cell_group=plot_cell_group,
    t_ms_offset=t_ms0,
    tick_ms_step=20,
    plot_t_end_ms=220,
    save_path=None,
    vmax=vmax,
)
plt.show()



#%% ############ Plot SVM results here ############
wp = r'E:\#Preprocessed_Data\260402_TC_Analysis'
brain_areas = ['MSB','ASB','AL','ML']
used_ares = brain_areas[2]

Decode_Frame_Timewin = pd.read_parquet(ot.Join(wp, f'{used_ares}_decode_svm_timewin_img20_win10_step5.parquet'))



train_shuffle = 0
test_shuffle = 4
x_ms_min, x_ms_max = 0, 250
y_ms_min, y_ms_max = 0, 250
vmax=1
# Use one group name (e.g. ['cat_1']) or many (e.g. ['all', 'cat_0', 'cat_1', 'cat_2'])
# plot_cell_groups = ['all', 'cat_0', 'cat_1', 'cat_2']
plot_cell_groups = ['cat_2']

for plot_cell_group in plot_cell_groups:
    plot_df = Decode_Frame_Timewin.loc[
        (Decode_Frame_Timewin['Cell_Group'] == str(plot_cell_group))
        &
        (Decode_Frame_Timewin['Train_Shuffle'] == int(train_shuffle))
        & (Decode_Frame_Timewin['Test_Shuffle'] == int(test_shuffle))
    ].copy()

    if plot_df.empty:
        print(f'Skip {plot_cell_group}: no rows for selected shuffle levels.')
        continue

    # Optional time-range crop for plotting (ms, based on window start).
    if x_ms_min is not None:
        plot_df = plot_df.loc[plot_df['Test_Window'] >= int(x_ms_min)]
    if x_ms_max is not None:
        plot_df = plot_df.loc[plot_df['Test_Window'] <= int(x_ms_max)]
    if y_ms_min is not None:
        plot_df = plot_df.loc[plot_df['Train_Window'] >= int(y_ms_min)]
    if y_ms_max is not None:
        plot_df = plot_df.loc[plot_df['Train_Window'] <= int(y_ms_max)]

    if plot_df.empty:
        print(f'Skip {plot_cell_group}: no rows remain after range filtering.')
        continue

    acc_mat = plot_df.pivot_table(
        index='Train_Window',
        columns='Test_Window',
        values='Accuracy',
        aggfunc='mean',
    ).sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        acc_mat,
        ax=ax,
        mask=acc_mat.isna(),
        vmin=0.0, center=0.05,
        vmax=vmax,
        square=True,
        cmap='coolwarm',
        cbar_kws={'label': 'Decoding Accuracy'},
    )
    # Draw diagonal (train window == test window) for visible range.
    row_labels = acc_mat.index.to_numpy(dtype=int)
    col_labels = acc_mat.columns.to_numpy(dtype=int)
    row_pos = {int(v): i for i, v in enumerate(row_labels)}
    col_pos = {int(v): j for j, v in enumerate(col_labels)}
    common_starts = sorted(set(row_pos.keys()) & set(col_pos.keys()))
    if common_starts:
        xs = [col_pos[v] + 0.5 for v in common_starts]
        ys = [row_pos[v] + 0.5 for v in common_starts]
        ax.plot(
            xs,
            ys,
            color='gray',
            linewidth=1.2,
            linestyle='--',
            alpha=0.8,
            zorder=5,
            clip_on=True,
        )

    ax.set_xlabel('Test Time Window Start (ms)')
    ax.set_ylabel('Train Time Window Start (ms)')
    ax.set_title(
        f'{used_ares} ({plot_cell_group}) SVM decode heatmap\n'
        f'Train shuffle={train_shuffle}, Test shuffle={test_shuffle}\n'
        f'X:[{x_ms_min},{x_ms_max}] ms, Y:[{y_ms_min},{y_ms_max}] ms'
    )
    ax.invert_yaxis()  # Revert the y axis
    plt.tight_layout()
    plt.show()

#%%
# CV accuracy curves across train windows for each shuffle level (0-4).
plot_cell_group = 'all'
cv_curve_df = (
    Decode_Frame_Timewin.loc[
        Decode_Frame_Timewin['Cell_Group'] == str(plot_cell_group),
        ['Train_Shuffle', 'Train_Window', 'CV_Accuracy'],
    ]
    .drop_duplicates()
    .groupby(['Train_Shuffle', 'Train_Window'], as_index=False)['CV_Accuracy']
    .mean()
    .sort_values(['Train_Shuffle', 'Train_Window'])
)

fig, ax = plt.subplots(figsize=(8, 5))
palette = sns.color_palette('tab10', n_colors=5)
for sh in range(5):
    sub = cv_curve_df.loc[cv_curve_df['Train_Shuffle'] == sh]
    if sub.empty:
        continue
    ax.plot(
        sub['Train_Window'].to_numpy(dtype=float),
        sub['CV_Accuracy'].to_numpy(dtype=float),
        linewidth=2.0,
        color=palette[sh],
        label=f'Shuffle {sh}',
    )

ax.set_xlabel('Time Window Start (ms)')
ax.set_ylabel('CV Accuracy')
ax.set_title(f'{used_ares} ({plot_cell_group}) CV decoding across time windows')
ax.set_ylim(0.0, 1.1)
ax.grid(alpha=0.25, linestyle='--', linewidth=0.8)
ax.legend(frameon=False, ncol=1, title='Train shuffle')
plt.tight_layout()
plt.show()

#%%

