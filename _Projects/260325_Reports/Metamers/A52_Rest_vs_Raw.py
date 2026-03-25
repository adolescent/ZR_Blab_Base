'''
Estimate rest-part effect using the same logic as A51_Bubble_vs_Raw.
'''

#%%
import OS_Tools as ot
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from Py_Structure.Info_Files.InfoLoader import Load_Info

warnings.filterwarnings('ignore')

wp = r'E:\#Preprocessed_Data\260305_Report_Data\Bubbles'
datafoler = r'E:\#Preprocessed_Data\Selected_Cells'
brain_sites = ['AL', 'ASB', 'ML', 'MSB']
all_rsp = {}
raw_img_path = ot.Get_File_Name(r'E:\#Stimsets\Raw_Objects', '.jpg')[:20]

for site in tqdm(brain_sites, total=len(brain_sites), desc='Loading responses'):
    all_rsp[site] = np.load(ot.Join(datafoler, f'{site}_Cells_Bubble.npz'), allow_pickle=True)['psth']
    all_rsp[site] = all_rsp[site][:, :, 160:320].sum(-1)
    all_rsp[site] = all_rsp[site] / all_rsp[site].max(1, keepdims=True)

tsv_info, masks, raw_mask_file = Load_Info(setname='Metamer_Singlebubble_v251107', load_mask=True)
stim_seq = tsv_info.loc[300:].reset_index(drop=True)
mask_seq = masks[300:, :, :]

# trial index definition in 4200 timeline
metamers = np.arange(0, 1000)
bubbles = np.arange(1040, 2640)
rests = np.arange(2640, 4240)

# metamer condition table (1000 rows)
img_indices = np.tile(np.arange(1, 41), 25)
shuffle_levels = np.tile(np.repeat(np.arange(5), 40), 5)
metamer_conditions = pd.DataFrame({'Img_Index': img_indices, 'Shuffle_Level': shuffle_levels})


#%%
def build_rest_vs_raw_table(
    brain_sites: list[str] | None = None,
    img_ids: list[int] | None = None,
    bottom_k_rest: int = 10,
    save_path: str | None = None
) -> pd.DataFrame:
    """
    For each site x cell x img_id:
    - raw_mean_s0: mean response on metamer S0 (img 0..19 only)
    - least_rest_rsp: mean of bottom-k least activated rest trials
    - least_rest_trial_indices: IDs of selected rest trials (k entries)
    """
    if brain_sites is None:
        brain_sites = list(all_rsp.keys())
    if img_ids is None:
        img_ids = list(range(20))
    img_ids = [int(i) for i in img_ids]
    if bottom_k_rest < 1:
        raise ValueError(f'bottom_k_rest must be >= 1, got {bottom_k_rest}')

    metamer_df = metamer_conditions.copy()
    metamer_df['Trial_Index'] = np.arange(len(metamer_df))
    # align with rest IDs (0..19 only)
    metamer_df['Graph_ID'] = metamer_df['Img_Index'] - 1
    metamer_df = metamer_df[metamer_df['Graph_ID'].between(0, 19)].copy()

    rest_df = stim_seq.loc[rests, ['Object']].copy()
    rest_df['Trial_Index'] = rests
    rest_df['Graph_ID'] = rest_df['Object'].astype(int) - 1

    records = []
    for site in tqdm(brain_sites, total=len(brain_sites), desc='Computing rest-vs-raw'):
        if site not in all_rsp:
            raise ValueError(f'Unknown site {site}. Available: {list(all_rsp.keys())}')
        site_rsp = all_rsp[site]
        n_cells = site_rsp.shape[0]

        for graph_id in img_ids:
            raw_trials = metamer_df[
                (metamer_df['Graph_ID'] == graph_id) &
                (metamer_df['Shuffle_Level'] == 0)
            ]['Trial_Index'].to_numpy(dtype=int)
            rest_trials = rest_df[rest_df['Graph_ID'] == graph_id]['Trial_Index'].to_numpy(dtype=int)

            # expected shape check: raw=5, rest=80
            if len(raw_trials) != 5 or len(rest_trials) != 80:
                continue

            raw_mean = site_rsp[:, raw_trials].mean(axis=1)
            rest_mat = site_rsp[:, rest_trials]  # N_Cell x 80
            k = min(bottom_k_rest, rest_mat.shape[1])

            # bottom-k selection (least activated)
            bottomk_pos_unsorted = np.argpartition(rest_mat, k - 1, axis=1)[:, :k]
            bottomk_vals = np.take_along_axis(rest_mat, bottomk_pos_unsorted, axis=1)
            least_rest_rsp = bottomk_vals.mean(axis=1)

            # sort selected trials by response ascending for traceability
            row_order = np.argsort(bottomk_vals, axis=1)
            bottomk_pos = np.take_along_axis(bottomk_pos_unsorted, row_order, axis=1)
            bottomk_trials = rest_trials[bottomk_pos]

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(raw_mean != 0, least_rest_rsp / raw_mean, np.nan)

            for cell_id in range(n_cells):
                records.append({
                    'Site': site,
                    'Cell_ID': int(cell_id),
                    'Img_ID': int(graph_id),
                    'BottomK_Rest_N': int(k),
                    'Raw_Mean_S0': float(raw_mean[cell_id]),
                    'Least_Rest_Rsp': float(least_rest_rsp[cell_id]),
                    'Least_Rest_over_RawMean': float(ratio[cell_id]),
                    'Least_Rest_Trial_Indices': bottomk_trials[cell_id].astype(int).tolist(),
                })

    rest_df_out = pd.DataFrame(records)
    if save_path is not None:
        rest_df_out.to_parquet(save_path, index=False)
    return rest_df_out


def read_important_mask_from_rest(
    rest_df: pd.DataFrame,
    site: str,
    img_id: int,
    cell_id: int,
    return_mask_only: bool = False,
    show: bool = True
) -> np.ndarray:
    """
    Build avg least-rest mask -> reverse it -> return important masked image.
    """
    row = rest_df[
        (rest_df['Site'] == site) &
        (rest_df['Img_ID'] == int(img_id)) &
        (rest_df['Cell_ID'] == int(cell_id))
    ]
    if len(row) == 0:
        raise ValueError(f'No row found for Site={site}, Img_ID={img_id}, Cell_ID={cell_id}.')
    if len(row) > 1:
        raise ValueError(f'Multiple rows found for Site={site}, Img_ID={img_id}, Cell_ID={cell_id}.')

    rest_trials = [int(i) for i in row.iloc[0]['Least_Rest_Trial_Indices']]
    if len(rest_trials) == 0:
        raise ValueError('Least_Rest_Trial_Indices is empty.')

    avg_rest_mask = np.mean(np.stack([mask_seq[t].astype(np.float32) for t in rest_trials], axis=0), axis=0)
    important_mask = np.clip(1.0 - avg_rest_mask, 0, 1)

    raw_img = np.array(Image.open(raw_img_path[int(img_id)]).convert('RGB'), dtype=np.float32)
    important_img = np.round(raw_img * important_mask[..., None]).astype(np.uint8)

    if show:
        plt.figure(figsize=(4, 4), dpi=120)
        if return_mask_only:
            plt.imshow(important_mask, cmap='gray', vmin=0, vmax=1)
            plt.title(f'Important mask | {site} | img {img_id} | cell {cell_id}')
        else:
            plt.imshow(important_img)
            plt.title(f'Important image | {site} | img {img_id} | cell {cell_id}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return important_mask if return_mask_only else important_img


def estimate_rest_roi(
    best_rest_df: pd.DataFrame,
    thres: float = 0.5,
    save_path: str | None = None
) -> pd.DataFrame:
    """
    Estimate mask size (ROI) of reversed rest mask for each row.

    - Build avg least-rest mask from Least_Rest_Trial_Indices
    - Reverse to important mask: 1 - avg_rest_mask
    - Binary by threshold
    - Compute ON proportion
    """
    if not (0 <= thres <= 1):
        raise ValueError(f'thres should be in [0,1], got {thres}')

    records = []
    for rid, row in tqdm(best_rest_df.iterrows(), total=len(best_rest_df), desc='Estimating rest ROI'):
        rest_trials = [int(i) for i in row['Least_Rest_Trial_Indices']]
        if len(rest_trials) == 0:
            continue

        avg_rest_mask = np.mean(
            np.stack([mask_seq[t].astype(np.float32) for t in rest_trials], axis=0),
            axis=0
        )
        important_mask = np.clip(1.0 - avg_rest_mask, 0, 1)
        bin_mask = (important_mask >= thres).astype(np.uint8)

        n_total = int(bin_mask.size)
        n_on = int(bin_mask.sum())

        records.append({
            'Source_Row': int(rid),
            'Site': row['Site'],
            'Cell_ID': int(row['Cell_ID']),
            'Img_ID': int(row['Img_ID']),
            'Img_Index': int(row['Img_ID']),
            'BottomK_Rest_N': int(row['BottomK_Rest_N']),
            'Least_Rest_Trial_Indices': rest_trials,
            'Threshold': float(thres),
            'Important_Mask_Mean': float(important_mask.mean()),
            'ROI_On_Pixels': n_on,
            'ROI_Total_Pixels': n_total,
            'ROI_On_Ratio': float(n_on / n_total),
        })

    rest_roi_df = pd.DataFrame(records)
    if save_path is not None:
        rest_roi_df.to_parquet(save_path, index=False)
    return rest_roi_df


#%% run
best_rest_df = build_rest_vs_raw_table(
    brain_sites=brain_sites,
    img_ids=list(range(20)),
    bottom_k_rest=10,
    save_path=ot.Join(wp, 'Best_Rest_vs_Raw.parquet')
)

plt.hist(best_rest_df.Least_Rest_over_RawMean,bins=np.linspace(0,1.5,50))
#%%
_ = read_important_mask_from_rest(
    best_rest_df,
    site='AL',
    img_id=11,
    cell_id=210,
    return_mask_only=False,
    show=True
)



#%%
rest_roi_df = estimate_rest_roi(
    best_rest_df=best_rest_df,
    thres=0.5,
    save_path=ot.Join(wp, 'Best_Rest_ROI.parquet')
)

#%% Boxplot: rest ROI ratio by brain area and image index
plt.figure(figsize=(10, 5), dpi=120)
sns.boxplot(
    data=rest_roi_df,
    x='Site',
    y='ROI_On_Ratio',
    hue='Img_Index',
    order=['MSB', 'ML', 'ASB', 'AL'],
    showfliers=False,
    width=0.5
)
plt.xlabel('Brain Area')
plt.ylabel('ROI_On_Ratio')
plt.tight_layout()
plt.show()

#%%
### Compare Bubble and rest mask, calculate overlapping ratio. Based on Bubble and Rest seperately.

def estimate_bubble_rest_overlap(
    best_bubble_df: pd.DataFrame,
    best_rest_df: pd.DataFrame,
    thres: float = 0.5,
    save_path: str | None = None
) -> pd.DataFrame:
    """
    For each matched Site x Cell_ID x Img_ID:
    - thresholded bubble mask
    - thresholded rest-important mask (1 - avg least-rest mask)
    - overlap ratios toward bubble and toward rest-important masks
    """
    if not (0 <= thres <= 1):
        raise ValueError(f'thres should be in [0,1], got {thres}')

    key_cols = ['Site', 'Cell_ID', 'Img_ID']
    merged = pd.merge(
        best_bubble_df[key_cols + ['Best_Bubble_Trial_Indices', 'TopK_Bubble_N']],
        best_rest_df[key_cols + ['Least_Rest_Trial_Indices', 'BottomK_Rest_N']],
        on=key_cols,
        how='inner'
    )

    records = []
    for rid, row in tqdm(merged.iterrows(), total=len(merged), desc='Estimating bubble-rest overlap'):
        bubble_trials = [int(i) for i in row['Best_Bubble_Trial_Indices']]
        rest_trials = [int(i) for i in row['Least_Rest_Trial_Indices']]
        if len(bubble_trials) == 0 or len(rest_trials) == 0:
            continue

        bubble_avg = np.mean(
            np.stack([mask_seq[t].astype(np.float32) for t in bubble_trials], axis=0),
            axis=0
        )
        rest_avg = np.mean(
            np.stack([mask_seq[t].astype(np.float32) for t in rest_trials], axis=0),
            axis=0
        )
        rest_important = np.clip(1.0 - rest_avg, 0, 1)

        bubble_bin = bubble_avg >= thres
        rest_imp_bin = rest_important >= thres
        inter = bubble_bin & rest_imp_bin
        union = bubble_bin | rest_imp_bin

        bubble_on = int(bubble_bin.sum())
        rest_on = int(rest_imp_bin.sum())
        inter_on = int(inter.sum())
        union_on = int(union.sum())

        overlap_toward_bubble = np.nan if bubble_on == 0 else inter_on / bubble_on
        overlap_toward_rest = np.nan if rest_on == 0 else inter_on / rest_on
        iou = np.nan if union_on == 0 else inter_on / union_on

        records.append({
            'Source_Row': int(rid),
            'Site': row['Site'],
            'Cell_ID': int(row['Cell_ID']),
            'Img_ID': int(row['Img_ID']),
            'Threshold': float(thres),
            'TopK_Bubble_N': int(row['TopK_Bubble_N']),
            'BottomK_Rest_N': int(row['BottomK_Rest_N']),
            'Bubble_On_Pixels': bubble_on,
            'RestImportant_On_Pixels': rest_on,
            'Overlap_On_Pixels': inter_on,
            'Overlap_Ratio_to_Bubble': float(overlap_toward_bubble),
            'Overlap_Ratio_to_RestImportant': float(overlap_toward_rest),
            'Overlap_IoU': float(iou),
        })

    overlap_df = pd.DataFrame(records)
    if save_path is not None:
        overlap_df.to_parquet(save_path, index=False)
    return overlap_df


#%% load bubble table and compute overlap with rest table
bubble_df_path = ot.Join(wp, 'Best_Bubble_vs_Raw.parquet')
best_bubble_df = pd.read_parquet(bubble_df_path)

overlap_df = estimate_bubble_rest_overlap(
    best_bubble_df=best_bubble_df,
    best_rest_df=best_rest_df,
    thres=0.5,
    save_path=ot.Join(wp, 'Bubble_Rest_Mask_Overlap.parquet')
)

#%% overlap plot by brain area (hue: metric type)
overlap_plot_df = overlap_df[['Site', 'Overlap_Ratio_to_Bubble', 'Overlap_Ratio_to_RestImportant', 'Overlap_IoU']].copy()
overlap_plot_df = overlap_plot_df.melt(
    id_vars='Site',
    value_vars=['Overlap_Ratio_to_Bubble', 'Overlap_Ratio_to_RestImportant', 'Overlap_IoU'],
    var_name='Overlap_Type',
    value_name='Overlap_Value'
).dropna()
overlap_plot_df['Overlap_Type'] = overlap_plot_df['Overlap_Type'].map({
    'Overlap_Ratio_to_Bubble': 'to bubble',
    'Overlap_Ratio_to_RestImportant': 'to rest',
    'Overlap_IoU': 'IoU'
})

plt.figure(figsize=(6, 5), dpi=120)
sns.boxplot(
    data=overlap_plot_df,
    x='Site',
    y='Overlap_Value',
    hue='Overlap_Type',
    order=['MSB', 'ML', 'ASB', 'AL'],
    showfliers=False,
    width=0.6
)
plt.xlabel('Brain Area')
plt.ylabel('Overlap Ratio')
plt.tight_layout()
plt.show()


