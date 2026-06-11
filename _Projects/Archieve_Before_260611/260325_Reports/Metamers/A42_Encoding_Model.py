'''
This script will train a encoding model to encode the stimulus from DCNN middle layers.

'''



#%%
import itertools
import seaborn as sns
import OS_Tools as ot
from PIL import Image
import numpy as np
from Matrix_Tools import Corr_Matrix
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

wp = r'E:\#Preprocessed_Data\260305_Report_Data\Decoding_SVM_MetamerOnly'
datafoler = r'E:\#Preprocessed_Data\Selected_Cells'
brain_sites = ['AL','ASB','ML','MSB']
dcnn_sites = ['Alex','VGG16','Res50']
dcnn_conv_layers = ['last_conv','last_conv','last_conv']
dcnn_fc_layers = ['fc6','fc1','avgpool']
# dcnn_files = [r'Alex_Response.npz',r'VGG16_Response.npz',r'Res50_Response.npz']


avr_neuro_resp = {}
avr_neuro_celing_index = {}
avr_dcnn_conv_resp = {}
avr_dcnn_fc_resp = {}

for site,conv_layer,fc_layer in zip(dcnn_sites,dcnn_conv_layers,dcnn_fc_layers):
    data = np.load(ot.Join(datafoler,f'{site}_Response.npz'),allow_pickle=True)
    avr_dcnn_conv_resp[site] = data[conv_layer].reshape(1000,-1)
    avr_dcnn_fc_resp[site] = data[fc_layer]

# load average response for each site.
for site in tqdm(brain_sites):
    data = np.load(ot.Join(datafoler,f'{site}_Cells_Metamer_Only.npz'),allow_pickle=True)
    c_rsp = data['psth'][:,:,160:320].sum(-1)
    z_rsp = (c_rsp-c_rsp.mean(1,keepdims=True))/c_rsp.std(1,keepdims=True)
    z_rsp = np.clip(z_rsp,-10,10)
    avr_neuro_resp[site] = z_rsp.T
    avr_neuro_celing_index[site] = data['ceiling_index']
# response here are (1000,N_dim),encoding can be done from here.

#%%
img_indices = np.tile(np.arange(1, 41), 25)
# Generate Shuffle Level column: for each of 0-4, repeat 40 times; the whole 0-4 block is repeated 25 times
shuffle_levels = np.tile(np.repeat(np.arange(5), 40), 5)
# Ensure arrays are the correct length
assert len(img_indices) == 1000
assert len(shuffle_levels) == 1000
df_conditions = pd.DataFrame({
    'Img_Index': img_indices,
    'Shuffle_Level': shuffle_levels
})



#%%


def run_encoding_model(
    brain_site,
    dcnn_site,
    dcnn_layer,
    train_indices,
    eval_indices,
    pca_variance_explained,
    *,
    ridge_alpha=1.0,
    standardize_before_pca=True,
    ceiling_eps=1e-6,
    avr_neuro_resp_map=None,
    avr_neuro_celing_index_map=None,
    avr_dcnn_conv_resp_map=None,
    avr_dcnn_fc_resp_map=None,
):
    """
    DCNN (X) -> neuron responses (Y) encoding with PCA + Ridge.

    dcnn_layer: 'conv' or 'fc'.
    train_indices / eval_indices: row indices into the 1000 trials (0..999).
    pca_variance_explained: float in (0, 1], e.g. 0.8 or 0.95 for retained variance.
    R = Pearson r between predicted and observed on eval trials; R_adj = R / max(sqrt(ceiling), eps).

    Optional *_map args default to module-level dicts loaded above.
    """
    if avr_neuro_resp_map is None:
        avr_neuro_resp_map = avr_neuro_resp
    if avr_neuro_celing_index_map is None:
        avr_neuro_celing_index_map = avr_neuro_celing_index
    if avr_dcnn_conv_resp_map is None:
        avr_dcnn_conv_resp_map = avr_dcnn_conv_resp
    if avr_dcnn_fc_resp_map is None:
        avr_dcnn_fc_resp_map = avr_dcnn_fc_resp

    if dcnn_layer == 'conv':
        X_row = avr_dcnn_conv_resp_map[dcnn_site]
    elif dcnn_layer == 'fc':
        X_row = avr_dcnn_fc_resp_map[dcnn_site]
    else:
        raise ValueError("dcnn_layer must be 'conv' or 'fc'")

    Y_row = avr_neuro_resp_map[brain_site]
    ceiling = np.asarray(avr_neuro_celing_index_map[brain_site], dtype=float)

    assert X_row.shape[0] == 1000 and Y_row.shape[0] == 1000
    assert Y_row.shape[1] == ceiling.shape[0]

    train_indices = np.asarray(train_indices, dtype=int)
    eval_indices = np.asarray(eval_indices, dtype=int)
    assert train_indices.ndim == 1 and eval_indices.ndim == 1
    assert len(train_indices) > 0 and len(eval_indices) > 0

    X_train = X_row[train_indices]
    Y_train = Y_row[train_indices]
    X_eval = X_row[eval_indices]
    Y_eval = Y_row[eval_indices]

    if standardize_before_pca:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)
    else:
        scaler = None

    pca = PCA(n_components=pca_variance_explained, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train)
    X_eval_pca = pca.transform(X_eval)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train_pca, Y_train)
    Y_pred = model.predict(X_eval_pca)

    n_cells = Y_eval.shape[1]
    r = np.full(n_cells, np.nan)
    for j in range(n_cells):
        y_true = Y_eval[:, j]
        y_hat = Y_pred[:, j]
        if np.std(y_true) == 0 or np.std(y_hat) == 0:
            continue
        r[j], _ = pearsonr(y_true, y_hat)

    denom = np.sqrt(np.maximum(ceiling, 0.0))
    denom = np.maximum(denom, ceiling_eps)
    R_adj = r / denom

    return {
        'r': r,
        'R_adj': R_adj,
        'Y_pred': Y_pred,
        'Y_eval': Y_eval,
        'n_train': len(train_indices),
        'n_eval': len(eval_indices),
        'n_pca_components': int(pca.n_components_),
        'pca_variance_explained_total': float(np.sum(pca.explained_variance_ratio_)),
        'scaler': scaler,
        'pca': pca,
        'model': model,
    }


def trial_indices_for_shuffle_ani(df, shuffle_level, ani):
    """
    ani: 1 = animate (Img_Index 1..20), 0 = inanimate (Img_Index 21..40).
    Returns row indices into the 1000 trials for that shuffle × category.
    """
    if ani == 1:
        mask = (df['Shuffle_Level'] == shuffle_level) & (df['Img_Index'] <= 20)
    elif ani == 0:
        mask = (df['Shuffle_Level'] == shuffle_level) & (df['Img_Index'] > 20)
    else:
        raise ValueError('ani must be 0 (inanimate) or 1 (animate)')
    return df.loc[mask].index.to_numpy()


def build_trial_index_lookup(df):
    """10 conditions: 5 shuffle levels × 2 (animate / inanimate)."""
    lookup = {}
    for shuffle_level, ani in itertools.product(range(5), (0, 1)):
        idx = trial_indices_for_shuffle_ani(df, shuffle_level, ani)
        lookup[(shuffle_level, ani)] = idx
    return lookup


def build_encoding_results_dataframe(
    trial_idx_lookup,
    *,
    dcnn_sites_list=None,
    brain_sites_list=None,
    dcnn_layers=('conv', 'fc'),
    pca_variance_explained=0.8,
    ridge_alpha=1.0,
    standardize_before_pca=True,
):
    """
    Full grid: each (DCNN site × conv/fc) × brain area × 10 train × 10 test conditions,
    one row per cell with R_adj.

    Backbone column is '{dcnn_site}_{layer}' (e.g. Alex_conv, VGG16_fc).

    Columns: Backbone, Brain_Area, Train_Ani, Test_Ani, Train_Shuffle, Test_Shuffle,
             Cell, R_raw, R_adj
    """
    if dcnn_sites_list is None:
        dcnn_sites_list = dcnn_sites
    if brain_sites_list is None:
        brain_sites_list = brain_sites

    shuffle_ani_pairs = list(itertools.product(range(5), (0, 1)))
    rows = []
    train_test_pairs = list(itertools.product(shuffle_ani_pairs, shuffle_ani_pairs))

    outer_total = len(dcnn_sites_list) * len(dcnn_layers) * len(brain_sites_list)
    for dcnn_site, dcnn_layer, brain_area in tqdm(
        itertools.product(dcnn_sites_list, dcnn_layers, brain_sites_list),
        total=outer_total,
        desc='Site×Layer×Area',
    ):
        backbone_label = f'{dcnn_site}_{dcnn_layer}'
        n_cells = avr_neuro_resp[brain_area].shape[1]
        for (train_sh, train_ani), (test_sh, test_ani) in tqdm(
            train_test_pairs,
            total=len(train_test_pairs),
            leave=False,
            desc='10×10 sh×ani',
        ):
            train_indices = trial_idx_lookup[(train_sh, train_ani)]
            test_indices = trial_idx_lookup[(test_sh, test_ani)]
            res = run_encoding_model(
                brain_area,
                dcnn_site,
                dcnn_layer,
                train_indices,
                test_indices,
                pca_variance_explained,
                ridge_alpha=ridge_alpha,
                standardize_before_pca=standardize_before_pca,
            )
            r_raw = res['r']
            r_adj = res['R_adj']
            for cell_idx in range(n_cells):
                rows.append({
                    'Backbone': backbone_label,
                    'Brain_Area': brain_area,
                    'Train_Ani': train_ani,
                    'Test_Ani': test_ani,
                    'Train_Shuffle': train_sh,
                    'Test_Shuffle': test_sh,
                    'Cell': cell_idx,
                    'R_raw': r_raw[cell_idx],
                    'R_adj': r_adj[cell_idx],
                })

    out = pd.DataFrame(rows)
    cols = ['Backbone', 'Brain_Area', 'Train_Ani', 'Test_Ani', 'Train_Shuffle', 'Test_Shuffle', 'Cell', 'R_raw', 'R_adj']
    return out[cols]


#%%
savepath = r'E:\#Preprocessed_Data\260305_Report_Data\Encoding_Model'
trial_idx_lookup = build_trial_index_lookup(df_conditions)

encoding_results_df = build_encoding_results_dataframe(
    trial_idx_lookup,
    dcnn_layers=('conv', 'fc'),
    pca_variance_explained=0.8,
    ridge_alpha=1.0,
    standardize_before_pca=True,
)

encoding_results_df.to_parquet(
    ot.Join(savepath, 'encoding_grid_results.parquet'),
    index=False,
    engine='pyarrow',
)
print(encoding_results_df.head())
print(len(encoding_results_df))


#%%
### Visualization part
def plot_encoding_rraw_heatmaps(
    encoding_df,
    selected_backbones,
    selected_brain_areas,
    *,
    aggfunc='mean',
    figsize_scale=3.8,
    cmap='viridis',
    vmin=None,
    vmax=None,
):
    """
    Plot 10x10 heatmaps of R_raw for selected Backbone x Brain_Area groups.

    Axis order for both train (X) and test (Y):
    shuffle0-4 animate, then shuffle0-4 inanimate.
    """
    if isinstance(selected_backbones, str):
        selected_backbones = [selected_backbones]
    if isinstance(selected_brain_areas, str):
        selected_brain_areas = [selected_brain_areas]

    cond_order = [(s, 1) for s in range(5)] + [(s, 0) for s in range(5)]
    cond_ticks = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    cond_to_idx = {k: i for i, k in enumerate(cond_order)}

    n_row = len(selected_brain_areas)
    n_col = len(selected_backbones)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(figsize_scale * n_col, figsize_scale * n_row),
        squeeze=False,
    )
    # Leave room on the right for one shared colorbar.
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
    first_hm = None

    for i, area in enumerate(selected_brain_areas):
        for j, backbone in enumerate(selected_backbones):
            ax = axes[i, j]
            sub = encoding_df[
                (encoding_df['Backbone'] == backbone) &
                (encoding_df['Brain_Area'] == area)
            ]

            mat = np.full((10, 10), np.nan)
            if not sub.empty:
                grouped = sub.groupby(
                    ['Train_Shuffle', 'Train_Ani', 'Test_Shuffle', 'Test_Ani'],
                    as_index=False
                )['R_adj'].agg(aggfunc)
                for _, row in grouped.iterrows():
                    y_key = (int(row['Train_Shuffle']), int(row['Train_Ani']))
                    x_key = (int(row['Test_Shuffle']), int(row['Test_Ani']))
                    if y_key in cond_to_idx and x_key in cond_to_idx:
                        mat[cond_to_idx[y_key], cond_to_idx[x_key]] = row['R_adj']

            hm = sns.heatmap(
                mat,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                xticklabels=cond_ticks,
                yticklabels=cond_ticks,
                square=True,
                cbar=(first_hm is None),
                cbar_ax=cbar_ax if first_hm is None else None,
            )
            if first_hm is None:
                first_hm = hm
                cbar_ax.set_ylabel('R_adj', rotation=90)
            ax.set_title(f'{backbone} | {area}')
            ax.set_xlabel('Test (Shuffle)')
            ax.set_ylabel('Train (Shuffle)')
            ax.axvline(5, color='white', lw=1.5, alpha=0.7)
            ax.axhline(5, color='white', lw=1.5, alpha=0.7)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

    # Show Ani/Inani grouping text only once for the whole figure.
    fig.text(0.46, 0.04, 'Test groups: Ani (0-4) | Inani (0-4)', ha='center')
    fig.text(0.03, 0.50, 'Train groups: Ani (0-4) | Inani (0-4)', va='center', rotation=90)
    plt.tight_layout(rect=[0.05, 0.06, 0.90, 0.98])
    return fig, axes


def build_triangle_stats_dataframe(encoding_df, value_col='R_adj'):
    """
    Extract lower/upper triangle values from ani-ani and inani-inani blocks.

    Lower/Upper are defined by shuffle index relation:
    - lower: Train_Shuffle > Test_Shuffle
    - upper: Train_Shuffle < Test_Shuffle
    Diagonal is excluded.
    """
    df = encoding_df.copy()
    bb = df['Backbone'].str.rsplit('_', n=1, expand=True)
    df['Backbone_Base'] = bb[0]
    df['Layer'] = bb[1]

    same_block = df['Train_Ani'] == df['Test_Ani']
    off_diag = df['Train_Shuffle'] != df['Test_Shuffle']
    tri_df = df[same_block & off_diag].copy()
    tri_df['Block'] = np.where(tri_df['Train_Ani'] == 1, 'Ani', 'Inani')
    tri_df['Triangle'] = np.where(
        tri_df['Train_Shuffle'] > tri_df['Test_Shuffle'],
        'Lower',
        'Upper',
    )
    tri_df = tri_df.rename(columns={value_col: 'corr_adj'})
    tri_df['Hue'] = tri_df['Backbone_Base'] + '_' + tri_df['Triangle']
    keep_cols = [
        'Backbone',
        'Backbone_Base',
        'Layer',
        'Brain_Area',
        'Block',
        'Triangle',
        'Cell',
        'corr_adj',
        'Hue',
    ]
    return tri_df[keep_cols]


def plot_triangle_stats_boxplot(triangle_df, figsize=(16, 6)):
    """
    Boxplot of lower vs upper triangle stats.
    X: Brain_Area, Y: corr_adj, Hue: Backbone_Base + Triangle
    Two subplots: conv and fc.
    """
    layers = ['conv', 'fc']
    x_order = ['MSB', 'ML', 'ASB', 'AL']

    # Fixed backbone order for legend/hue consistency.
    preferred_backbones = ['Alex', 'VGG16', 'Res50']
    available_backbones = [b for b in preferred_backbones if b in set(triangle_df['Backbone_Base'])]
    if len(available_backbones) == 0:
        available_backbones = sorted(triangle_df['Backbone_Base'].unique().tolist())

    # Hue order: all lower first, then all upper.
    hue_order = (
        [f'{b}_Lower' for b in available_backbones] +
        [f'{b}_Upper' for b in available_backbones]
    )

    # Use darker color for Lower and lighter paired color for Upper.
    lower_palette = sns.color_palette('tab10', n_colors=max(3, len(available_backbones)))
    palette = {}
    for i, b in enumerate(available_backbones):
        c = lower_palette[i]
        lighter = tuple(min(1.0, 0.45 + 0.55 * ch) for ch in c)
        palette[f'{b}_Lower'] = c
        palette[f'{b}_Upper'] = lighter

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharey=True)

    for ax, layer in zip(axes, layers):
        sub = triangle_df[triangle_df['Layer'] == layer]
        sns.boxplot(
            data=sub,
            x='Brain_Area',
            y='corr_adj',
            hue='Hue',
            order=x_order,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            showfliers=False,whis=(5,95)
        )
        ax.set_title(f'{layer.upper()}')
        ax.set_xlabel('Brain area')
        ax.set_ylabel('corr_adj')
        ax.set_ylim(-0.2,1.2)
        ax.tick_params(axis='x', rotation=0)
        if layer == 'fc' and ax.get_legend() is not None:
            ax.get_legend().remove()
        if layer == 'conv':
            ax.legend(title='Backbone_Triangle', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    return fig, axes


# Example:
fig, axes = plot_encoding_rraw_heatmaps(
    encoding_results_df,
    # selected_backbones=['Alex_conv', 'Alex_fc', 'Res50_conv', 'Res50_fc', 'VGG16_conv', 'VGG16_fc'],
    selected_backbones=['VGG16_conv','VGG16_fc'],
    # selected_backbones=['Alex_conv','Alex_fc'],
    # selected_brain_areas=['ML','MSB'],
    selected_brain_areas=['AL','ASB'],
    aggfunc='mean',
    cmap='magma',vmax=1,vmin=0
)
plt.show()

#%%
# Triangle stats + boxplot example
triangle_stats_df = build_triangle_stats_dataframe(encoding_results_df, value_col='R_adj')
fig, axes = plot_triangle_stats_boxplot(triangle_stats_df, figsize=(7, 12))
plt.show()

#%%




