'''
Use SVM to decode in odd one out task, follow steps as :

1. Pair data into triangle
2. Use response decode odd one
3. Compare change level
4. In different img and different area.
'''

#%%
import seaborn as sns
import OS_Tools as ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from itertools import combinations
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from Py_Structure.Info_Files.Stim_ID_List import Stim_ID


datafolder=r'E:\#Preprocessed_Data\Selected_Cells'
savepath = r'E:\#Preprocessed_Data\260305_Report_Data\Site_ANOVAs'

# filename = r'Res50_Response.npz'
# filename = r'Alex_Response.npz'

brain_area = 'MSB'
filename = fr'{brain_area}_Cells_Metamer_Only.npz'

data = np.load(ot.Join(datafolder,filename))
keys = list(data.keys())
n_img = 1000
avr_rsp = data['psth'][:,:,160:320].sum(-1)
normed_avr_rsp = avr_rsp/avr_rsp.max(1,keepdims=True)
zscored_rsp = (avr_rsp-avr_rsp.mean(1,keepdims=True))/avr_rsp.std(1,keepdims=True)
zscored_rsp = np.clip(zscored_rsp,-10,10)

del data

metamer_infos = Stim_ID('Metamer_Raw').Stim_Conditions

#%%
def decode_shuffle_0_vs_4_per_img(zscored_rsp, metamer_infos):
    """
    Decode shuffle level 0 vs 4 for each Img_Index independently.

    - Uses only rows where Shuffle_Level is 0 or 4.
    - Maps labels: 0 -> class 0, 4 -> class 1.
    - Runs Leave-One-Out CV with fold-wise scaling to avoid data leakage.
    """
    valid_mask = metamer_infos['Shuffle_Level'].isin([0, 4]).values
    selected_rsp = zscored_rsp[:, valid_mask]
    selected_info = metamer_infos.loc[valid_mask].reset_index(drop=True)

    img_indices = np.sort(selected_info['Img_Index'].unique())
    loo = LeaveOneOut()
    trial_records = []

    for img_index in tqdm(img_indices, desc='Decoding per Img_Index'):
        img_mask = (selected_info['Img_Index'] == img_index).values
        img_rsp = selected_rsp[:, img_mask]  # shape: (n_cells, n_trials_for_this_img)
        img_levels = selected_info.loc[img_mask, 'Shuffle_Level'].values

        # sklearn expects (n_samples, n_features)
        x = img_rsp.T
        y = (img_levels == 4).astype(int)

        # Safety checks for binary decoding.
        if (x.shape[0] < 2) or (np.unique(y).size < 2):
            continue

        # StandardScaler is fit only on each training fold in CV.
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', probability=True)
        )
        # Out-of-fold probability for each sample; column index 1 corresponds to class label 1.
        probas = cross_val_predict(clf, x, y, cv=loo, method='predict_proba')
        p_shuffle4 = probas[:, 1]

        img_df = pd.DataFrame({
            'Img_Index': int(img_index),
            'Shuffle_Level': img_levels.astype(int),
            'Label': y.astype(int),
            'P_Shuffle4': p_shuffle4.astype(float),
            'Pred_Label': (p_shuffle4 >= 0.5).astype(int),
        })
        trial_records.append(img_df)

    trial_prob_df = pd.concat(trial_records, axis=0, ignore_index=True)
    img_summary_df = (
        trial_prob_df
        .groupby('Img_Index', as_index=False)
        .agg(
            n_samples=('Label', 'size'),
            mean_p_shuffle4=('P_Shuffle4', 'mean'),
            mean_p_when_label0=('P_Shuffle4', lambda s: s[trial_prob_df.loc[s.index, 'Label'] == 0].mean()),
            mean_p_when_label1=('P_Shuffle4', lambda s: s[trial_prob_df.loc[s.index, 'Label'] == 1].mean()),
        )
        .sort_values('Img_Index')
        .reset_index(drop=True)
    )
    return trial_prob_df, img_summary_df


def pairwise_pearson_distance_per_img(rsp, metamer_infos):
    """
    Compute pair-wise Pearson distance (1-r) for shuffle pairs 0-0, 0-4, and 4-4.
    """
    valid_mask = metamer_infos['Shuffle_Level'].isin([0, 4]).values
    selected_rsp = rsp[:, valid_mask]
    selected_info = metamer_infos.loc[valid_mask].reset_index(drop=True)
    selected_original_idx = np.where(valid_mask)[0]

    img_indices = np.sort(selected_info['Img_Index'].unique())
    pair_records = []

    for img_index in tqdm(img_indices, desc='Pairwise Pearson per Img_Index'):
        img_mask = (selected_info['Img_Index'] == img_index).values
        img_rsp = selected_rsp[:, img_mask]  # (n_cells, n_trials_for_img)
        img_levels = selected_info.loc[img_mask, 'Shuffle_Level'].values.astype(int)
        img_original_idx = selected_original_idx[img_mask]

        n_trials = img_rsp.shape[1]
        if n_trials < 2:
            continue

        for i, j in combinations(range(n_trials), 2):
            level_i = int(img_levels[i])
            level_j = int(img_levels[j])
            pair_key = tuple(sorted((level_i, level_j)))
            if pair_key not in [(0, 0), (0, 4), (4, 4)]:
                continue

            vec_i = img_rsp[:, i]
            vec_j = img_rsp[:, j]
            pearson_r = np.corrcoef(vec_i, vec_j)[0, 1]
            pearson_dist = 1 - pearson_r

            pair_records.append({
                'Img_Index': int(img_index),
                'Pair_Type': f'{pair_key[0]}-{pair_key[1]}',
                'Trial_i_OriginalIndex': int(img_original_idx[i]),
                'Trial_j_OriginalIndex': int(img_original_idx[j]),
                'Shuffle_i': level_i,
                'Shuffle_j': level_j,
                'Pearson_r': float(pearson_r),
                'Pearson_Distance': float(pearson_dist),
            })

    pairwise_df = pd.DataFrame(pair_records)
    return pairwise_df


def plot_pairwise_pearson_by_type(pairwise_df, img_set=None, ax=None, title_suffix=''):
    """
    Visualize Pearson distance grouped by Pair_Type.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Output from pairwise_pearson_distance_per_img.
    img_set : list-like or None
        Optional image set to include, e.g. [1, 2, 3]. None means all images.
    ax : matplotlib axis or None
        If None, create a new figure.
    title_suffix : str
        Optional string appended to plot title.
    """
    plot_df = pairwise_df.copy()
    if img_set is not None:
        img_set = np.array(img_set).astype(int)
        plot_df = plot_df[plot_df['Img_Index'].isin(img_set)].copy()

    if plot_df.empty:
        raise ValueError('No data left for plotting. Check img_set selection.')

    pair_order = ['0-0', '0-4', '4-4']
    plot_df['Pair_Type'] = pd.Categorical(plot_df['Pair_Type'], categories=pair_order, ordered=True)
    plot_df = plot_df.sort_values('Pair_Type')

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4), dpi=150)

    sns.violinplot(
        data=plot_df,
        x='Pair_Type',
        y='Pearson_Distance',
        order=pair_order,
        inner='box',        # 或 None
        cut=0,
        bw_adjust=1.5,      # 增大平滑，避免太尖
        scale='width',
        linewidth=1.2,width=0.5,
        palette=['#4C72B0', '#55A868', '#C44E52'],
        ax=ax
    )
    sns.stripplot(
        data=plot_df,
        x='Pair_Type',
        y='Pearson_Distance',
        order=pair_order,
        color='black',
        alpha=0.2,
        size=2,
        jitter=0.2,
        ax=ax
    )

    n_imgs = plot_df['Img_Index'].nunique()
    ax.set_xlabel('Pair Type')
    ax.set_ylabel('Pearson Distance (1-r)')
    ax.set_title(f'Pair-wise Pearson Distance by Pair Type\nN images = {n_imgs}{title_suffix}')
    ax.grid(axis='y', alpha=0.25)
    return ax

#%%
if __name__ == '__main__':

    dcnn_rsp = np.load(ot.Join(r'E:\#Preprocessed_Data\Selected_Cells','Alex_Response_conv5_unpooled.npz'))['conv5_unpooled'].reshape(1000,-1).T
    trial_prob_df, img_summary_df = decode_shuffle_0_vs_4_per_img(dcnn_rsp, metamer_infos)
    pairwise_df = pairwise_pearson_distance_per_img(dcnn_rsp, metamer_infos)

    # Pairwise distance summary per image and pair type.
    pairwise_summary_df = (
        pairwise_df
        .groupby(['Img_Index', 'Pair_Type'], as_index=False)
        .agg(
            n_pairs=('Pearson_Distance', 'size'),
            mean_distance=('Pearson_Distance', 'mean'),
            std_distance=('Pearson_Distance', 'std'),
        )
    )

    # os.makedirs(savepath, exist_ok=True)
    # trial_pkl = ot.Join(savepath, f'{brain_area}_SVM_LOO_Trial_Prob.pkl')
    # trial_csv = ot.Join(savepath, f'{brain_area}_SVM_LOO_Trial_Prob.csv')
    # summary_pkl = ot.Join(savepath, f'{brain_area}_SVM_LOO_Img_Summary.pkl')
    # summary_csv = ot.Join(savepath, f'{brain_area}_SVM_LOO_Img_Summary.csv')

    # trial_prob_df.to_pickle(trial_pkl)
    # trial_prob_df.to_csv(trial_csv, index=False)
    # img_summary_df.to_pickle(summary_pkl)
    # img_summary_df.to_csv(summary_csv, index=False)

    # print('\nLeave-One-Out probability decoding finished.')
    # print(f'Trial-level rows: {len(trial_prob_df)}')
    # print(f'Image-level rows: {len(img_summary_df)}')
    # print(f'Saved: {trial_pkl}')
    # print(f'Saved: {summary_pkl}')
    # print(trial_prob_df.head())
    print(pairwise_df.head())
    print(pairwise_summary_df.head())

    # Optional image set selection for plotting; set None for all images.
    img_set = None
    # Example: img_set = [1, 2, 3, 4, 5]
    ax = plot_pairwise_pearson_by_type(pairwise_df, img_set=img_set)
    plt.tight_layout()
    plt.show()

#%% ######## Data Demo ###############
def _sample_random_pair_distances(points_a, points_b=None, n_pairs=5000, rng=None):
    """Sample Euclidean distances from random point pairs."""
    if rng is None:
        rng = np.random.default_rng(0)

    points_a = np.asarray(points_a)
    if points_b is None:
        # Within-set distances, avoid pairing a point with itself.
        idx_a = rng.integers(0, points_a.shape[0], size=n_pairs)
        idx_b = rng.integers(0, points_a.shape[0], size=n_pairs)
        same_mask = (idx_a == idx_b)
        while np.any(same_mask):
            idx_b[same_mask] = rng.integers(0, points_a.shape[0], size=np.sum(same_mask))
            same_mask = (idx_a == idx_b)
        vec = points_a[idx_a] - points_a[idx_b]
    else:
        points_b = np.asarray(points_b)
        idx_a = rng.integers(0, points_a.shape[0], size=n_pairs)
        idx_b = rng.integers(0, points_b.shape[0], size=n_pairs)
        vec = points_a[idx_a] - points_b[idx_b]

    return np.sqrt((vec ** 2).sum(axis=1))


def generate_2d_distance_matched_demo(
    n_blue=400,
    n_red=400,
    n_pairs=10000,
    seed=42,
):
    """
    Generate 2D red/blue data where:
    centroid distance(red-blue) ~= mean distance(blue-blue),
    while blue has larger within-distribution variation.
    """
    rng = np.random.default_rng(seed)

    # Blue: larger spread (especially on y-axis).
    blue_cov = np.array([[0.25, 0.0], [0.0, 3.2]])
    # Red: tighter cluster.
    red_cov = np.array([[0.18, 0.02], [0.02, 0.25]])

    blue_points = rng.multivariate_normal(mean=[0.0, 0.0], cov=blue_cov, size=n_blue)
    red_noise = rng.multivariate_normal(mean=[0.0, 0.0], cov=red_cov, size=n_red)

    blue_blue_dists = _sample_random_pair_distances(blue_points, None, n_pairs=n_pairs, rng=rng)
    target_centroid_dist = blue_blue_dists.mean()

    # Place red centroid on +x so centroid distance matches blue internal distance scale.
    red_shift = np.array([target_centroid_dist, 0.0])
    red_points = red_noise + red_shift
    red_blue_dists = _sample_random_pair_distances(red_points, blue_points, n_pairs=n_pairs, rng=rng)

    demo_df = pd.DataFrame(
        np.vstack([red_points, blue_points]),
        columns=['x', 'y']
    )
    demo_df['Group'] = (['Red'] * n_red) + (['Blue'] * n_blue)

    # Simple separability indicator on x-axis (large means easier linear separation).
    blue_x = blue_points[:, 0]
    red_x = red_points[:, 0]
    sep_x = (red_x.mean() - blue_x.mean()) / np.sqrt(red_x.var() + blue_x.var())
    centroid_dist = float(np.linalg.norm(red_points.mean(axis=0) - blue_points.mean(axis=0)))

    stats = pd.DataFrame({
        'Metric': [
            'centroid_distance_red_blue',
            'mean_distance_blue_blue',
            'mean_distance_red_blue',
            'blue_var_x',
            'blue_var_y',
            'red_var_x',
            'red_var_y',
            'x_axis_separability_index',
        ],
        'Value': [
            centroid_dist,
            float(blue_blue_dists.mean()),
            float(red_blue_dists.mean()),
            float(blue_x.var()),
            float(blue_points[:, 1].var()),
            float(red_x.var()),
            float(red_points[:, 1].var()),
            float(sep_x),
        ]
    })
    return demo_df, stats, red_blue_dists, blue_blue_dists


def plot_2d_distance_matched_demo(demo_df, red_blue_dists, blue_blue_dists):
    """Visualize 2D points and distance distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    ax0 = axes[0]
    red_df = demo_df[demo_df['Group'] == 'Red']
    blue_df = demo_df[demo_df['Group'] == 'Blue']

    ax0.scatter(blue_df['x'], blue_df['y'], s=14, c='royalblue', alpha=0.45, label='Blue')
    ax0.scatter(red_df['x'], red_df['y'], s=14, c='crimson', alpha=0.45, label='Red')
    blue_center = blue_df[['x', 'y']].mean().values
    red_center = red_df[['x', 'y']].mean().values
    ax0.scatter(blue_center[0], blue_center[1], s=110, c='navy', marker='X', label='Blue centroid')
    ax0.scatter(red_center[0], red_center[1], s=110, c='darkred', marker='X', label='Red centroid')
    ax0.set_title('2D demo points')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.axis('equal')
    ax0.legend(frameon=False, fontsize=8)

    ax1 = axes[1]
    centroid_dist = float(np.linalg.norm(red_center - blue_center))
    dist_df = pd.DataFrame({
        'Distance': np.concatenate([red_blue_dists, blue_blue_dists]),
        'Type': (['Red-Blue'] * len(red_blue_dists)) + (['Blue-Blue'] * len(blue_blue_dists)),
    })
    sns.kdeplot(data=dist_df, x='Distance', hue='Type', fill=True, alpha=0.3, ax=ax1)
    ax1.axvline(centroid_dist, color='black', linestyle='--', linewidth=1.6, label='Centroid distance')
    ax1.set_title('Distance distributions')
    ax1.set_xlabel('Euclidean distance')
    ax1.set_ylabel('Density')
    ax1.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    return fig, axes


if __name__ == '__main__':
    demo_df, demo_stats, rb_dists, bb_dists = generate_2d_distance_matched_demo(
        n_blue=450,
        n_red=450,
        n_pairs=12000,
        seed=93
    )
    print('\n2D distance-matched demo stats:')
    print(demo_stats)
    plot_2d_distance_matched_demo(demo_df, rb_dists, bb_dists)
    plt.show()
