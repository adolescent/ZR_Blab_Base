'''
This script will try to compare the bubble vs raw response, and see whether bubble is enough for activation.

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
from Py_Structure.Info_Files.InfoLoader import Load_Info
warnings.filterwarnings('ignore')

wp = r'E:\#Preprocessed_Data\260305_Report_Data\Bubbles'
datafoler = r'E:\#Preprocessed_Data\Selected_Cells'
brain_sites = ['AL','ASB','ML','MSB']
all_bubble_rsp ={}
raw_img_path = ot.Get_File_Name(r'E:\#Stimsets\Raw_Objects','.jpg')[:20]


for site in tqdm(brain_sites,total=len(brain_sites)):
    all_bubble_rsp[site] = np.load(ot.Join(datafoler,f'{site}_Cells_Bubble.npz'),allow_pickle=True)['psth']
    all_bubble_rsp[site] = all_bubble_rsp[site][:,:,160:320].sum(-1)
    all_bubble_rsp[site] = all_bubble_rsp[site]/all_bubble_rsp[site].max(1,keepdims=True)


tsv_info,masks,raw_mask_file = Load_Info(setname=f'Metamer_Singlebubble_v251107',load_mask=True)
stim_seq = tsv_info.loc[300:].reset_index(drop=True)
mask_seq = masks[300:,:,:]

#%% id infox.
metamers = np.arange(0,1000)
bubbles = np.arange(1040,2640)
rests = np.arange(2640,4240)
img_indices = np.tile(np.arange(1, 41), 25)
# Generate Shuffle Level column: for each of 0-4, repeat 40 times; the whole 0-4 block is repeated 25 times
shuffle_levels = np.tile(np.repeat(np.arange(5), 40), 5)
# Ensure arrays are the correct length
assert len(img_indices) == 1000
assert len(shuffle_levels) == 1000
metamer_conditions = pd.DataFrame({
    'Img_Index': img_indices,
    'Shuffle_Level': shuffle_levels
})



#%%
def build_bubble_vs_raw_table(
    brain_sites: list[str] | None = None,
    img_ids: list[int] | None = None,
    top_k_bubbles: int = 1,
    save_path: str | None = None
) -> pd.DataFrame:
    """
    For each site x cell x img_id:
    - raw_mean_s0: mean metamer S0 response
    - raw_max_s0: max metamer S0 response
    - best_bubble_rsp: mean response of top-k bubble trials (k=top_k_bubbles)
    - best_ratio_bubble_over_raw: best_bubble_rsp / raw_mean_s0
    - best_ratio_bubble_over_rawmax: best_bubble_rsp / raw_max_s0
    Also keeps winning bubble trial indices.
    """
    if brain_sites is None:
        brain_sites = list(all_bubble_rsp.keys())
    if img_ids is None:
        img_ids = list(range(20))
    img_ids = [int(i) for i in img_ids]
    if top_k_bubbles < 1:
        raise ValueError(f'top_k_bubbles must be >= 1, got {top_k_bubbles}')

    # Prepare trial-index mapping tables
    metamer_df = metamer_conditions.copy()
    metamer_df['Trial_Index'] = np.arange(len(metamer_df))
    # IMPORTANT: only keep metamer image IDs 0..19 (Img_Index 1..20), i.e. 500 trials.
    metamer_df['Graph_ID'] = metamer_df['Img_Index'] - 1
    metamer_df = metamer_df[metamer_df['Graph_ID'].between(0, 19)].copy()

    bubble_df = stim_seq.loc[bubbles, ['Object']].copy()
    bubble_df['Trial_Index'] = bubbles
    bubble_df['Graph_ID'] = bubble_df['Object'].astype(int) - 1

    records = []
    for site in tqdm(brain_sites, total=len(brain_sites), desc='Computing bubble-vs-raw'):
        if site not in all_bubble_rsp:
            raise ValueError(f'Unknown site {site}. Available: {list(all_bubble_rsp.keys())}')

        site_rsp = all_bubble_rsp[site]  # shape: N_Cell x 4240
        n_cells = site_rsp.shape[0]

        for graph_id in img_ids:
            # Raw baseline from metamer shuffle-0 for this graph
            raw_trials = metamer_df[
                (metamer_df['Graph_ID'] == graph_id) &
                (metamer_df['Shuffle_Level'] == 0)
            ]['Trial_Index'].to_numpy(dtype=int)
            # Bubble candidates for this graph
            bubble_trials = bubble_df[bubble_df['Graph_ID'] == graph_id]['Trial_Index'].to_numpy(dtype=int)

            if len(raw_trials) == 0 or len(bubble_trials) == 0:
                continue
            # Expected design for selected Img_ID 0..19
            if len(raw_trials) != 5 or len(bubble_trials) != 80:
                continue

            raw_mean = site_rsp[:, raw_trials].mean(axis=1)          # (N_Cell,)
            raw_max = site_rsp[:, raw_trials].max(axis=1)            # (N_Cell,)
            bubble_mat = site_rsp[:, bubble_trials]                  # (N_Cell, N_BubbleTrial)
            k = min(top_k_bubbles, bubble_mat.shape[1])
            # Get top-k indices for each cell without full sort
            topk_pos_unsorted = np.argpartition(bubble_mat, -k, axis=1)[:, -k:]   # (N_Cell, k)
            topk_vals = np.take_along_axis(bubble_mat, topk_pos_unsorted, axis=1)  # (N_Cell, k)
            best_bubble = topk_vals.mean(axis=1)                                     # (N_Cell,)

            # For traceability, store top-k trials sorted by response descending
            row_order = np.argsort(topk_vals, axis=1)[:, ::-1]
            topk_pos = np.take_along_axis(topk_pos_unsorted, row_order, axis=1)
            topk_trials = bubble_trials[topk_pos]                                    # (N_Cell, k)

            with np.errstate(divide='ignore', invalid='ignore'):
                best_ratio = np.where(raw_mean != 0, best_bubble / raw_mean, np.nan)
                best_ratio_rawmax = np.where(raw_max != 0, best_bubble / raw_max, np.nan)

            for cell_id in range(n_cells):
                trial_idx_list = topk_trials[cell_id].astype(int).tolist()

                records.append({
                    'Site': site,
                    'Cell_ID': int(cell_id),
                    'Img_ID': int(graph_id),
                    'Raw_Mean_S0': float(raw_mean[cell_id]),
                    'Raw_Max_S0': float(raw_max[cell_id]),
                    'Best_Bubble_Rsp': float(best_bubble[cell_id]),
                    'Best_Ratio_Bubble_over_Raw': float(best_ratio[cell_id]),
                    'Best_Ratio_Bubble_over_RawMax': float(best_ratio_rawmax[cell_id]),
                    'TopK_Bubble_N': int(k),
                    'Best_Bubble_Trial_Indices': trial_idx_list,
                })

    result_df = pd.DataFrame(records)

    if save_path is not None:
        result_df.to_parquet(save_path, index=False)

    return result_df


def generate_masked_graph_from_row(df: pd.DataFrame, row_id: int, show: bool = True) -> np.ndarray:
    """
    Generate weighted masked graph on-demand from one dataframe row.
    Opacity is mean of masks from Best_Bubble_Trial_Indices.
    """
    row = df.loc[row_id]
    img_id = int(row['Img_ID'])
    trial_idx_list = [int(i) for i in row['Best_Bubble_Trial_Indices']]

    if img_id < 0 or img_id >= len(raw_img_path):
        raise ValueError(f'Img_ID {img_id} is out of raw_img_path range [0, {len(raw_img_path)-1}]')
    if len(trial_idx_list) == 0:
        raise ValueError('Best_Bubble_Trial_Indices is empty, cannot generate masked graph.')

    c_masks = np.stack([mask_seq[t].astype(np.float32) for t in trial_idx_list], axis=0)  # (k, H, W)
    alpha = np.clip(c_masks.mean(axis=0), 0, 1)[..., None]                                  # (H, W, 1)
    raw_img = np.array(Image.open(raw_img_path[img_id]).convert('RGB'), dtype=np.float32)   # (H, W, 3)
    img_array = np.round(raw_img * alpha).astype(np.uint8)

    if show:
        plt.figure(figsize=(4, 4), dpi=120)
        plt.imshow(img_array)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return img_array


def read_masked_graph(
    df: pd.DataFrame,
    site: str,
    img_id: int,
    cell_id: int,
    return_mask_only: bool = False,
    show: bool = True
) -> np.ndarray:
    """
    Return masked image for one site/img/cell row.
    If return_mask_only=True, return the averaged mask instead.
    """
    c_rows = df[
        (df['Site'] == site) &
        (df['Img_ID'] == int(img_id)) &
        (df['Cell_ID'] == int(cell_id))
    ]
    if len(c_rows) == 0:
        raise ValueError(f'No row found for Site={site}, Img_ID={img_id}, Cell_ID={cell_id}.')
    if len(c_rows) > 1:
        raise ValueError(f'Multiple rows found for Site={site}, Img_ID={img_id}, Cell_ID={cell_id}.')

    trial_idx_list = [int(i) for i in c_rows.iloc[0]['Best_Bubble_Trial_Indices']]
    if len(trial_idx_list) == 0:
        raise ValueError('Best_Bubble_Trial_Indices is empty, cannot generate averaged mask.')

    avg_mask = np.mean(
        np.stack([mask_seq[t].astype(np.float32) for t in trial_idx_list], axis=0),
        axis=0
    )

    raw_img = np.array(Image.open(raw_img_path[int(img_id)]).convert('RGB'), dtype=np.float32)
    masked_img = np.round(raw_img * avg_mask[..., None]).astype(np.uint8)

    if show:
        plt.figure(figsize=(4, 4), dpi=120)
        if return_mask_only:
            plt.imshow(avg_mask, cmap='gray', vmin=0, vmax=1)
            plt.title(f'Avg mask | {site} | img {img_id} | cell {cell_id}')
        else:
            plt.imshow(masked_img)
            plt.title(f'Masked image | {site} | img {img_id} | cell {cell_id}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return avg_mask if return_mask_only else masked_img


def estimate_roi_from_best_bubble(
    best_bubble_df: pd.DataFrame,
    thres: float = 0.5,
    save_path: str | None = None
) -> pd.DataFrame:
    """
    Estimate ROI for each row in best_bubble_df.

    Steps:
    1) Build averaged mask from Best_Bubble_Trial_Indices.
    2) Binarize by threshold (avg_mask >= thres).
    3) Compute ON-mask proportion.
    4) Save all key metadata and ROI stats.
    """
    if not (0 <= thres <= 1):
        raise ValueError(f'thres should be in [0,1], got {thres}')

    records = []
    for rid, row in tqdm(best_bubble_df.iterrows(), total=len(best_bubble_df), desc='Estimating ROI'):
        trial_idx_list = [int(i) for i in row['Best_Bubble_Trial_Indices']]
        if len(trial_idx_list) == 0:
            continue

        avg_mask = np.mean(
            np.stack([mask_seq[t].astype(np.float32) for t in trial_idx_list], axis=0),
            axis=0
        )
        bin_mask = (avg_mask >= thres).astype(np.uint8)

        n_total = int(bin_mask.size)
        n_on = int(bin_mask.sum())

        records.append({
            'Source_Row': int(rid),
            'Site': row['Site'],
            'Cell_ID': int(row['Cell_ID']),
            'Img_ID': int(row['Img_ID']),
            'TopK_Bubble_N': int(row['TopK_Bubble_N']),
            'Best_Bubble_Trial_Indices': trial_idx_list,
            'Threshold': float(thres),
            'Avg_Mask_Mean': float(avg_mask.mean()),
            'ROI_On_Pixels': n_on,
            'ROI_Total_Pixels': n_total,
            'ROI_On_Ratio': float(n_on / n_total),
        })

    roi_df = pd.DataFrame(records)
    if save_path is not None:
        roi_df.to_parquet(save_path, index=False)
    return roi_df


#%% Bubble-vs-Raw table (all sites)
best_bubble_df = build_bubble_vs_raw_table(
    brain_sites=brain_sites,
    img_ids=list(range(20)),
    top_k_bubbles=10,
    save_path=ot.Join(wp, 'Best_Bubble_vs_Raw.parquet')
)

plt.hist(best_bubble_df.Best_Ratio_Bubble_over_RawMax,bins=np.linspace(0,5,50))

#%%
_ = read_masked_graph(best_bubble_df, site='MSB', img_id=11, cell_id=21, show=True)

#%%
roi_df = estimate_roi_from_best_bubble(
    best_bubble_df=best_bubble_df,
    thres=0.5,
    save_path=ot.Join(wp, 'Best_Bubble_ROI.parquet')
)

#%% Boxplot: ROI ratio by brain area and image index
roi_plot_df = roi_df.copy()
roi_plot_df['Img_Index'] = roi_plot_df['Img_ID']

plt.figure(figsize=(10,5), dpi=120)
sns.boxplot(
    data=roi_plot_df,
    x='Site',
    y='ROI_On_Ratio',
    hue='Img_Index',
    # hue='Site',
    order=['MSB', 'ML', 'ASB', 'AL'],
    showfliers=False,width=0.5
)
plt.xlabel('Brain Area')
plt.ylabel('ROI_On_Ratio')
plt.ylim(0,0.25)
plt.tight_layout()
plt.show()
#%% visualize roi of different sites.

