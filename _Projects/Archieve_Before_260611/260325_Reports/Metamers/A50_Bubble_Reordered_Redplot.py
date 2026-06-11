'''
Plot reordered redplot of bubble vs raw.

'''


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
def build_reordered_index(
    stim_seq: pd.DataFrame,
    metamer_conditions: pd.DataFrame,
    graph_ids: list[int] | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build x-axis order as:
    Graph0[Raw-S1-S2-S3-S4, Bubbles, Rests], Graph1..., Graph19.

    - Raw-S1-S4 are from metamer set (first 1000 trials), grouped by Graph_ID (0-19).
    - Bubbles and Rests are grouped by Object ID with 80 trials per graph.
    """
    # Metamer block in response matrix: indices 0..999
    # IMPORTANT: only keep metamer image IDs 0..19 (Img_Index 1..20), i.e. 500 trials.
    metamer_df = metamer_conditions.copy()
    metamer_df['Trial_Index'] = np.arange(len(metamer_df))
    metamer_df['Graph_ID'] = metamer_df['Img_Index'] - 1
    metamer_df = metamer_df[metamer_df['Graph_ID'].between(0, 19)].copy()

    # Bubble and rest blocks from full 4200 response timeline
    bubble_df = stim_seq.loc[bubbles, ['Object']].copy()
    bubble_df['Trial_Index'] = bubbles
    bubble_df['Graph_ID'] = bubble_df['Object'].astype(int) - 1

    rest_df = stim_seq.loc[rests, ['Object']].copy()
    rest_df['Trial_Index'] = rests
    rest_df['Graph_ID'] = rest_df['Object'].astype(int) - 1

    if graph_ids is None:
        graph_ids = list(range(20))
    graph_ids = [int(g) for g in graph_ids]
    invalid_ids = [g for g in graph_ids if g < 0 or g > 19]
    if len(invalid_ids) > 0:
        raise ValueError(f'graph_ids must be in [0, 19], got invalid values: {invalid_ids}')

    ordered_indices = []
    trial_meta = []
    for graph_id in graph_ids:
        # Raw/S1/S2/S3/S4 in order (Shuffle_Level 0..4)
        for shuffle_level in range(5):
            c_block = metamer_df[
                (metamer_df['Graph_ID'] == graph_id) &
                (metamer_df['Shuffle_Level'] == shuffle_level)
            ]['Trial_Index'].to_numpy()
            ordered_indices.extend(c_block.tolist())
            trial_meta.extend([
                {'Graph_ID': graph_id, 'Block': f'S{shuffle_level}', 'Trial_Index': int(i)}
                for i in c_block
            ])

        # Bubbles (80 expected per graph)
        c_bubble = bubble_df[bubble_df['Graph_ID'] == graph_id]['Trial_Index'].to_numpy()
        ordered_indices.extend(c_bubble.tolist())
        trial_meta.extend([
            {'Graph_ID': graph_id, 'Block': 'Bubble', 'Trial_Index': int(i)}
            for i in c_bubble
        ])

        # Rests (80 expected per graph)
        c_rest = rest_df[rest_df['Graph_ID'] == graph_id]['Trial_Index'].to_numpy()
        ordered_indices.extend(c_rest.tolist())
        trial_meta.extend([
            {'Graph_ID': graph_id, 'Block': 'Rest', 'Trial_Index': int(i)}
            for i in c_rest
        ])

    ordered_indices = np.array(ordered_indices, dtype=int)
    trial_meta = pd.DataFrame(trial_meta)
    return ordered_indices, trial_meta


def plot_direct_response_image(
    site: str = 'AL',
    img_ids: list[int] | None = None,
    vmax_percentile: float = 99.0
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Return reordered matrix (N_Cell x 4200) and metadata, and plot direct response image.
    """
    if site not in all_bubble_rsp:
        raise ValueError(f'Unknown brain site: {site}. Choose from {list(all_bubble_rsp.keys())}')

    x_order, trial_meta = build_reordered_index(
        stim_seq=stim_seq,
        metamer_conditions=metamer_conditions,
        graph_ids=img_ids
    )
    c_rsp = all_bubble_rsp[site]  # shape: N_Cell x 4240
    reordered_rsp = c_rsp[:, x_order]

    # Per graph now: metamer 25 (S0..S4 each 5 reps for Img 0..19 only) + bubble 80 + rest 80 = 185
    trials_per_graph = 185
    expected_trials = trials_per_graph * (20 if img_ids is None else len(img_ids))
    if reordered_rsp.shape[1] != expected_trials:
        raise RuntimeError(
            f'Reordered trial count is {reordered_rsp.shape[1]}, expected {expected_trials}. '
            'Please check metamer/bubble/rest indexing.'
        )

    plt.figure(figsize=(8, 5), dpi=120)
    vmax = np.percentile(reordered_rsp, vmax_percentile)
    sns.heatmap(
        reordered_rsp,
        cmap='bwr',center=0,
        vmin=0,
        vmax=vmax,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    if img_ids is None:
        selected_graph_txt = 'Graph0..Graph19'
        n_graphs = 20
    else:
        selected_graph_txt = ','.join([str(i) for i in img_ids])
        n_graphs = len(img_ids)

    # plt.title(f'{site} direct response image (cells x reordered trials)')
    plt.xlabel(f'Re-ordered Response',fontsize=18)
    # plt.ylabel('Cell ID')

    # Visual separators:
    # - black lines between graphs (each graph has 210 trials)
    # - blue lines between metamer (S0-S4), bubble, and rest within each graph
    total_trials = trials_per_graph * n_graphs
    for graph_start in np.arange(0, total_trials, trials_per_graph):
        # 25 metamer trials (5 shuffle levels x 5 reps), then 80 bubble, then 80 rest
        plt.axvline(graph_start + 25, color='blue', lw=1.0, alpha=0.9)
        plt.axvline(graph_start + 105, color='blue', lw=1.0, alpha=0.9)

    for boundary in np.arange(trials_per_graph, total_trials, trials_per_graph):
        plt.axvline(boundary, color='black', lw=1.6, alpha=1.0)
    plt.tight_layout()
    plt.show()

    return reordered_rsp, trial_meta


# first direct-response plot
selected_site = 'MSB'  # choose from ['AL','ASB','ML','MSB']
selected_imgs = [5,6,7,8,9]  # graph/img IDs in [0..19], set None to use all
response_matrix, response_meta = plot_direct_response_image(site=selected_site, img_ids=selected_imgs)

#%% 

