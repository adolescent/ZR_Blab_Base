"""
MDS visualization for comparing shuffle level 0 vs 4 responses.
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import OS_Tools as ot
from sklearn.manifold import MDS
from Py_Structure.Info_Files.Stim_ID_List import Stim_ID
import copy


datafolder = r'E:\#Preprocessed_Data\Selected_Cells'
brain_area = 'MSB'
filename = fr'{brain_area}_Cells_Metamer_Only.npz'

data = np.load(ot.Join(datafolder, filename))
avr_rsp = data['psth'][:, :, 160:320].sum(-1)
zscored_rsp = (avr_rsp - avr_rsp.mean(1, keepdims=True)) / avr_rsp.std(1, keepdims=True)
zscored_rsp = np.clip(zscored_rsp, -10, 10)
del data

metamer_infos = Stim_ID('Metamer_Raw').Stim_Conditions


#%%
def prepare_img_level_response(response_matrix, meta_df, img_indices, levels=(0, 4)):
    """
    Pick one or multiple image indices and keep only specified shuffle levels.

    Returns
    -------
    selected_rsp : np.ndarray
        Shape (n_samples, n_neurons).
    selected_info : pd.DataFrame
        Metadata aligned to selected_rsp rows.
    """
    if np.isscalar(img_indices):
        img_indices = [int(img_indices)]
    else:
        img_indices = [int(v) for v in img_indices]

    valid_mask = (
        (meta_df['Img_Index'].isin(img_indices).values)
        & (meta_df['Shuffle_Level'].isin(levels).values)
    )
    selected_info = meta_df.loc[valid_mask].reset_index(drop=True)
    selected_rsp = response_matrix[:, valid_mask].T

    if selected_rsp.shape[0] == 0:
        raise ValueError(
            f'No samples for Img_Index in {img_indices} and Shuffle_Level in {levels}.'
        )
    if selected_rsp.shape[0] < 2:
        raise ValueError(
            f'Need >=2 samples for MDS, got {selected_rsp.shape[0]}.'
        )
    return selected_rsp, selected_info


def run_mds_2d(response_samples, random_state=42):
    """
    Project response vectors (samples x neurons) to 2D using metric MDS.
    """
    mds = MDS(
        n_components=2,
        metric=True,
        dissimilarity='euclidean',
        n_init=8,
        max_iter=600,
        random_state=random_state,
    )
    return mds.fit_transform(response_samples)


def build_unique_img_palette(unique_imgs, palette_name='husl'):
    """
    Build a categorical palette with unique colors for each image index.
    """
    colors = sns.color_palette(palette_name, n_colors=len(unique_imgs))
    return {img: colors[i] for i, img in enumerate(unique_imgs)}


def build_shuffle_markers(unique_levels):
    """
    Build marker map for arbitrary shuffle levels.
    Keep preferred mapping: 0 -> 'o', 4 -> 'X'.
    """
    marker_map = {}
    preferred =  {0: 'o',1:'s',2:'^',3:'d', 4: 'X'}
    fallback_markers = ['s', '^', 'D', 'P', 'v', '<', '>', '*', 'h', '8']

    for lv in unique_levels:
        if lv in preferred:
            marker_map[lv] = preferred[lv]

    fallback_idx = 0
    for lv in unique_levels:
        if lv in marker_map:
            continue
        marker_map[lv] = fallback_markers[fallback_idx % len(fallback_markers)]
        fallback_idx += 1
    return marker_map


def plot_mds_by_shuffle(
    mds_points,
    selected_info,
    ax=None,
    title_suffix='',
    cmap_name='husl',
    point_size=22,
):
    """
    Scatter plot for 2D MDS result:
    - color by Img_Index
    - marker by Shuffle_Level (0: circle, 4: x)
    """
    plot_df = selected_info.copy()
    plot_df['MDS1'] = mds_points[:, 0]
    plot_df['MDS2'] = mds_points[:, 1]
    plot_df['Shuffle_Level'] = plot_df['Shuffle_Level'].astype(int)
    plot_df['Img_Index'] = plot_df['Img_Index'].astype(int)
    unique_imgs = np.sort(plot_df['Img_Index'].unique())
    palette = build_unique_img_palette(unique_imgs, palette_name=cmap_name)
    unique_levels = np.sort(plot_df['Shuffle_Level'].unique())
    marker_map = build_shuffle_markers(unique_levels)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4), dpi=150)

    sns.scatterplot(
        data=plot_df,
        x='MDS1',
        y='MDS2',
        hue='Img_Index',
        style='Shuffle_Level',
        markers=marker_map,
        style_order=unique_levels,
        s=point_size,
        alpha=0.85,
        palette=palette,
        ax=ax,legend=False
    )
    ax.set_title(f'MDS | {title_suffix}')
    ax.grid(alpha=0.2)
    # ax.legend(frameon=False)
    return ax, plot_df


def compute_mds_dispersion_by_img_level(plot_df):
    """
    Compute dispersion in 2D MDS plane for each (Img_Index, Shuffle_Level).

    Dispersion metric:
        trace(covariance_matrix) = var(MDS1) + var(MDS2)
    """
    records = []
    group_keys = ['Img_Index', 'Shuffle_Level']

    grouped = plot_df.groupby(group_keys, sort=True)
    for (img_idx, shuffle_lv), sub_df in grouped:
        coords = sub_df[['MDS1', 'MDS2']].to_numpy(dtype=float)
        n_samples = coords.shape[0]

        if n_samples < 2:
            cov_trace = np.nan
            var_mds1 = np.nan
            var_mds2 = np.nan
        else:
            cov_mat = np.cov(coords, rowvar=False)
            cov_trace = float(np.trace(cov_mat))
            var_mds1 = float(cov_mat[0, 0])
            var_mds2 = float(cov_mat[1, 1])

        records.append({
            'Img_Index': int(img_idx),
            'Shuffle_Level': int(shuffle_lv),
            'n_samples': int(n_samples),
            'Dispersion_CovTrace': cov_trace,
            'Var_MDS1': var_mds1,
            'Var_MDS2': var_mds2,
        })

    dispersion_df = (
        plot_df[['Img_Index', 'Shuffle_Level']]
        .drop_duplicates()
        .merge(
            # Keep all groups sorted in final table.
            # records includes the computed dispersion metrics.
            pd.DataFrame(records),
            on=['Img_Index', 'Shuffle_Level'],
            how='left',
        )
        .sort_values(['Img_Index', 'Shuffle_Level'])
        .reset_index(drop=True)
    )
    return dispersion_df

#%%
if __name__ == '__main__':
    # 1) Choose image indices for MDS fitting and for visualization.
    selected_img_indices = list(range(1, 41))       # used to fit MDS
    visualized_img_indices = list(range(1, 41))     # subset shown in plots
    selected_levels = (0,1,2,3,4)  # e.g. (1, 4) or (0, 1, 2, 3, 4)
    selected_rsp, selected_info = prepare_img_level_response(
        response_matrix=zscored_rsp,
        meta_df=metamer_infos,
        img_indices=selected_img_indices,
        levels=selected_levels,
    )

    # 2) MDS to 2D.
    mds_points = run_mds_2d(selected_rsp, random_state=114514)

    # 3) Visualize only a subset of images using coordinates from full-fit MDS.
    vis_mask = selected_info['Img_Index'].isin(visualized_img_indices).values
    _, mds_df = plot_mds_by_shuffle(
        mds_points[vis_mask],
        selected_info.loc[vis_mask].reset_index(drop=True),
        title_suffix=(
            f' | MDS fit Img={selected_img_indices[0]}-{selected_img_indices[-1]}'
            f', shown Img={visualized_img_indices[0]}-{visualized_img_indices[-1]}'
        ),
        cmap_name='tab20',
        point_size=20,
    )
    plt.tight_layout()
    plt.show()

    print('Sample count by shuffle level:')
    print(mds_df['Shuffle_Level'].value_counts().sort_index())
    print('\nSample count by image x shuffle level:')
    print(mds_df.groupby(['Img_Index', 'Shuffle_Level']).size().unstack(fill_value=0))

    # 4) Dispersion in MDS plane: trace of covariance matrix.
    dispersion_df = compute_mds_dispersion_by_img_level(mds_df)
    print('\nMDS dispersion (trace of covariance) by image x shuffle level:')
    print(dispersion_df.head(20))
#%% visualize variance by shuffle level.
    # 5) Simple visualization of dispersion by shuffle level.
    plotable = dispersion_df.copy()
    plotable['Ani'] = plotable['Img_Index']<=20
    fig, ax = plt.subplots(figsize=(5.2, 4), dpi=150)
    sns.boxplot(

        data=plotable,
        x='Shuffle_Level',
        y='Dispersion_CovTrace',
        hue='Ani',
        dodge=True,
        width=0.55,
        linewidth=1.2,
        palette='Set2',showfliers=False,
        ax=ax,
    )
    # Keep the figure simple: hue is encoded by box color, so no legend needed.
    # leg = ax.get_legend()
    # if leg is not None:
    #     leg.remove()
    ax.set_xlabel('Shuffle Level')
    ax.set_ylabel('Dispersion (trace of covariance)')
    ax.set_title('MDS dispersion by shuffle level')
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()

#%%
