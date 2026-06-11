"""
PCA visualization for comparing shuffle level 0 vs 4 responses.
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import OS_Tools as ot
from itertools import combinations
from sklearn.decomposition import PCA
from Py_Structure.Info_Files.Stim_ID_List import Stim_ID


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
            f'Need >=2 samples for PCA, got {selected_rsp.shape[0]}.'
        )
    return selected_rsp, selected_info


def run_pca(response_samples, n_pcs_fit=10, random_state=42):
    """
    Fit PCA and return projected scores.

    Parameters
    ----------
    response_samples : np.ndarray
        Shape (n_samples, n_neurons).
    n_pcs_fit : int
        Number of PCs kept by PCA transform.
    """
    max_valid = min(response_samples.shape[0], response_samples.shape[1])
    n_pcs_fit = int(n_pcs_fit)
    if n_pcs_fit < 2:
        raise ValueError('n_pcs_fit must be >= 2.')
    if n_pcs_fit > max_valid:
        raise ValueError(f'n_pcs_fit={n_pcs_fit} exceeds max valid={max_valid}.')

    pca = PCA(n_components=n_pcs_fit, random_state=random_state)
    pc_scores = pca.fit_transform(response_samples)
    return pc_scores, pca


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
    preferred = {0: 'o',1:'s',2:'^',3:'d', 4: 'X'}
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


def plot_pca_pairwise(
    pc_scores,
    selected_info,
    vis_pcs=(1, 2, 3),
    title_suffix='',
    cmap_name='husl',
    point_size=20,
    vector_start_level=0,
    vector_end_level=4,
    draw_image_vectors=True,
):
    """
    Plot pairwise 2D projections from selected PCs.

    Parameters
    ----------
    pc_scores : np.ndarray
        Shape (n_samples, n_pcs_fit), output from run_pca.
    vis_pcs : tuple/list of int
        1-based PC indices used for plotting, e.g. (1,2,3).
    """
    vis_pcs = [int(v) for v in vis_pcs]
    if len(vis_pcs) < 2:
        raise ValueError('vis_pcs must contain at least 2 PCs.')
    if len(set(vis_pcs)) != len(vis_pcs):
        raise ValueError('vis_pcs contains duplicated PC index.')
    if min(vis_pcs) < 1:
        raise ValueError('vis_pcs should be 1-based positive indices.')
    if max(vis_pcs) > pc_scores.shape[1]:
        raise ValueError(
            f'Max vis_pcs={max(vis_pcs)} exceeds fitted PCs={pc_scores.shape[1]}.'
        )

    plot_df = selected_info.copy()
    for pc_idx in vis_pcs:
        plot_df[f'PC{pc_idx}'] = pc_scores[:, pc_idx - 1]
    plot_df['Shuffle_Level'] = plot_df['Shuffle_Level'].astype(int)
    plot_df['Img_Index'] = plot_df['Img_Index'].astype(int)

    unique_imgs = np.sort(plot_df['Img_Index'].unique())
    palette = build_unique_img_palette(unique_imgs, palette_name=cmap_name)
    unique_levels = np.sort(plot_df['Shuffle_Level'].unique())
    marker_map = build_shuffle_markers(unique_levels)

    pc_pairs = list(combinations(vis_pcs, 2))
    n_pairs = len(pc_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4.4), dpi=150)
    if n_pairs == 1:
        axes = [axes]

    # Per-image vector in PCA space: centroid(level_end) - centroid(level_start).
    pc_cols = [f'PC{pc_idx}' for pc_idx in vis_pcs]
    centroid_df = (
        plot_df.groupby(['Img_Index', 'Shuffle_Level'], as_index=False)[pc_cols].mean()
    )
    start_df = centroid_df[centroid_df['Shuffle_Level'] == int(vector_start_level)].copy()
    end_df = centroid_df[centroid_df['Shuffle_Level'] == int(vector_end_level)].copy()
    vector_df = start_df.merge(
        end_df,
        on='Img_Index',
        suffixes=('_start', '_end'),
        how='inner',
    )
    for pc_col in pc_cols:
        vector_df[f'{pc_col}_dx'] = vector_df[f'{pc_col}_end'] - vector_df[f'{pc_col}_start']

    vector_df = vector_df.sort_values('Img_Index').reset_index(drop=True)

    for ax, (pc_a, pc_b) in zip(axes, pc_pairs):
        x_col = f'PC{pc_a}'
        y_col = f'PC{pc_b}'
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue='Img_Index',
            style='Shuffle_Level',
            markers=marker_map,
            style_order=unique_levels,
            s=point_size,
            alpha=0.85,
            palette=palette,
            legend=False,
            edgecolor='none',
            ax=ax,
        )
        ax.set_title(f'{x_col} vs {y_col}')
        ax.grid(alpha=0.2)

        if draw_image_vectors and (len(vector_df) > 0):
            vec_colors = [palette[int(v)] for v in vector_df['Img_Index'].values]
            ax.quiver(
                vector_df[f'{x_col}_start'].to_numpy(dtype=float),
                vector_df[f'{y_col}_start'].to_numpy(dtype=float),
                vector_df[f'{x_col}_dx'].to_numpy(dtype=float),
                vector_df[f'{y_col}_dx'].to_numpy(dtype=float),
                angles='xy',
                scale_units='xy',
                scale=1,
                color=vec_colors,
                width=0.003,
                headwidth=3.8,
                headlength=5.5,
                alpha=0.8,
                zorder=6,
            )

    fig.suptitle(
        f'PCA pairwise projections | Color=Img_Index, Marker=Shuffle_Level{title_suffix}',
        y=1.02,
    )
    return fig, axes, plot_df


def compute_image_vectors_and_fit_r2(
    pc_scores,
    info_df,
    n_vector_dims=50,
    start_level=0,
    end_level=4,
):
    """
    Compute per-image vectors in PCA space (end-start) and fit R2 to a shared template.

    Shared template is the mean vector across images.
    For each image i:
        R2_i = ||proj(v_i on template)||^2 / ||v_i||^2
    Global fit R2:
        R2_global = sum_i ||proj(v_i)||^2 / sum_i ||v_i||^2
    """
    if n_vector_dims > pc_scores.shape[1]:
        raise ValueError(
            f'n_vector_dims={n_vector_dims} exceeds available PCs={pc_scores.shape[1]}.'
        )

    pc_cols = [f'PC{i}' for i in range(1, n_vector_dims + 1)]
    vec_df = info_df[['Img_Index', 'Shuffle_Level']].copy()
    for i, col in enumerate(pc_cols):
        vec_df[col] = pc_scores[:, i]
    vec_df['Img_Index'] = vec_df['Img_Index'].astype(int)
    vec_df['Shuffle_Level'] = vec_df['Shuffle_Level'].astype(int)

    centroid_df = vec_df.groupby(['Img_Index', 'Shuffle_Level'], as_index=False)[pc_cols].mean()
    start_df = centroid_df[centroid_df['Shuffle_Level'] == int(start_level)].copy()
    end_df = centroid_df[centroid_df['Shuffle_Level'] == int(end_level)].copy()
    merged = start_df.merge(end_df, on='Img_Index', suffixes=('_start', '_end'), how='inner')

    if merged.empty:
        raise ValueError(
            f'No image has both Shuffle_Level={start_level} and {end_level}.'
        )

    vector_cols = [f'Vec_PC{i}' for i in range(1, n_vector_dims + 1)]
    for i in range(1, n_vector_dims + 1):
        merged[f'Vec_PC{i}'] = merged[f'PC{i}_end'] - merged[f'PC{i}_start']

    v_mat = merged[vector_cols].to_numpy(dtype=float)
    template = v_mat.mean(axis=0)
    template_norm_sq = float(np.dot(template, template))
    vec_norm_sq = np.sum(v_mat ** 2, axis=1)

    if template_norm_sq < 1e-12:
        proj_norm_sq = np.zeros_like(vec_norm_sq)
        fit_r2_per_img = np.full_like(vec_norm_sq, np.nan, dtype=float)
        global_r2 = np.nan
    else:
        coeff = (v_mat @ template) / template_norm_sq
        proj = coeff[:, None] * template[None, :]
        proj_norm_sq = np.sum(proj ** 2, axis=1)
        fit_r2_per_img = np.divide(
            proj_norm_sq,
            vec_norm_sq,
            out=np.full_like(vec_norm_sq, np.nan, dtype=float),
            where=vec_norm_sq > 1e-12,
        )
        global_r2 = float(proj_norm_sq.sum() / vec_norm_sq.sum()) if vec_norm_sq.sum() > 1e-12 else np.nan

    out_df = merged[['Img_Index'] + vector_cols].copy()
    out_df['Vector_Norm'] = np.sqrt(vec_norm_sq)
    out_df['Fit_R2_to_Template'] = fit_r2_per_img
    out_df = out_df.sort_values('Img_Index').reset_index(drop=True)

    fit_summary = pd.DataFrame({
        'Metric': ['Template_Norm', 'Global_Fit_R2', 'N_Images'],
        'Value': [float(np.sqrt(template_norm_sq)), global_r2, int(len(out_df))],
    })
    return out_df, fit_summary


#%%
if __name__ == '__main__':
    # 1) Select image set for PCA fitting and for visualization.
    fit_img_indices = list(range(1, 41))         # fit PCA space on all 1000 images
    visualized_img_indices = list(range(21,41))     # only show this subset in plots
    selected_levels = (0,1,2,3,4)  # e.g. (0, 1, 2, 3, 4) or (1, 4)
    fit_rsp, fit_info = prepare_img_level_response(
        response_matrix=zscored_rsp,
        meta_df=metamer_infos,
        img_indices=fit_img_indices,
        levels=selected_levels,
    )

    # 2) PCA on all selected fitting images.
    n_pcs_fit = 50
    pc_scores, pca_model = run_pca(
        fit_rsp,
        n_pcs_fit=n_pcs_fit,
        random_state=42,
    )
    explained_var_df = pd.DataFrame({
        'PC': np.arange(1, n_pcs_fit + 1),
        'Explained_Var_Ratio': pca_model.explained_variance_ratio_,
        'Cumulative_Explained_Var': np.cumsum(pca_model.explained_variance_ratio_),
    })

    # 2.1) Build 50D vectors (0->4) and per-image fit R2 DataFrames.
    vector_50d_df, vector_fit_summary_df = compute_image_vectors_and_fit_r2(
        pc_scores=pc_scores,
        info_df=fit_info,
        n_vector_dims=50,
        start_level=0,
        end_level=4,
    )
    fit_r2_df = vector_50d_df[['Img_Index', 'Fit_R2_to_Template']].copy()
    vec_cols = [f'Vec_PC{i}' for i in range(1, 51)]
    vector_50d_df['Vector_Norm_50D'] = np.linalg.norm(
        vector_50d_df[vec_cols].to_numpy(dtype=float),
        axis=1,
    )
    img_vector_norm_df = vector_50d_df[['Img_Index', 'Vector_Norm_50D']].copy()

    # 3) Visualize only a subset of images in the fitted PCA space.
    vis_mask = fit_info['Img_Index'].isin(visualized_img_indices).values
    vis_info = fit_info.loc[vis_mask].reset_index(drop=True)
    vis_scores = pc_scores[vis_mask]
    if vis_scores.shape[0] == 0:
        raise ValueError('No samples left for visualization. Check visualized_img_indices.')

    # Visualization PCs (can be changed, 1-based).
    vis_pcs = (1, 2)  # e.g. (1,2) or (1,2,3)
    _, _, pca_df = plot_pca_pairwise(
        vis_scores,
        vis_info,
        vis_pcs=vis_pcs,
        title_suffix=(
            f' | PCA fit Img={fit_img_indices[0]}-{fit_img_indices[-1]}'
            f', shown Img={visualized_img_indices[0]}-{visualized_img_indices[-1]}'
        ),
        cmap_name='husl',
        point_size=10,
        vector_start_level=0,
        vector_end_level=4,
        draw_image_vectors=True,
    )
    plt.tight_layout()
    plt.show()

    print('Explained variance table of all fitted PCs:')
    print(explained_var_df)
    print('\nTotal explained variance (50 PCs):')
    print(np.round(explained_var_df['Explained_Var_Ratio'].sum(), 6))
    print('\nPer-image fit R2 (first 20 rows):')
    print(fit_r2_df.head(20))
    print('\nPer-image 50D vector norm (first 20 rows):')
    print(img_vector_norm_df.head(20))
    print('\nVector fit summary:')
    print(vector_fit_summary_df)
    print('\nSample count by shuffle level:')
    print(pca_df['Shuffle_Level'].value_counts().sort_index())
    # print('\nSample count by image x shuffle level:')
    # print(pca_df.groupby(['Img_Index', 'Shuffle_Level']).size().unstack(fill_value=0))
#%%

fit_r2_df['Ani'] = fit_r2_df['Img_Index']<=20
sns.histplot(data=fit_r2_df,x='Fit_R2_to_Template',hue='Ani',kde=True,bins=5)


