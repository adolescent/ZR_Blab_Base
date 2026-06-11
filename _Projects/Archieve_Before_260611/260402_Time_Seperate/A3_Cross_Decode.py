'''
Cut neuron response in time window, do 20 class svm decoding, test it's ability in or cross shuffle level.
'''

#%%
from Py_Structure.Info_Files.Stim_ID_List import Stim_ID
import OS_Tools as ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

Stim_Caller = Stim_ID(stim_type='Metamer_Raw')
metamer_ids = Stim_Caller.Stim_Conditions
datafolder = r'E:\#Preprocessed_Data\Selected_Cells'
wp= r'E:\#Preprocessed_Data\260402_TC_Analysis'

#%% data loading and pre-processing.
brain_areas = ['MSB','ASB','AL','ML']
used_ares = brain_areas[3]

psths = np.load(ot.Join(datafolder,f'{used_ares}_Cells_Metamer_Only.npz'))['psth']

# 0–300 ms → bins [100:400]; joint PMF over (img, time): denom = sum over 1000 imgs and 300 time pts
t_ms0, t_ms1 = 0, 300
t_idx0, t_idx1 = t_ms0 + 100, t_ms1 + 100
assert t_idx1 - t_idx0 == (t_ms1 - t_ms0)
resp_t = psths[:, :, t_idx0:t_idx1]
n_img, n_t = resp_t.shape[1], resp_t.shape[2]
den = resp_t.sum(axis=(1, 2), keepdims=True)
psth_pdf = np.divide(resp_t, den, out=np.zeros_like(resp_t, dtype=float), where=den > 0) * 1000

#%% Select cell sets['
cell_cats = np.load(ot.Join(wp,f'{used_ares}_pca_kmeans3_ztime_0to250ms.npz'))['labels']
if int(cell_cats.shape[0]) != int(psth_pdf.shape[0]):
    raise ValueError('cell_cats length does not match number of cells in psth_pdf.')

#%%
'''
PARAMETERS OF DECODER
'''
img_range = np.arange(1,21)
win_size = 10
win_step = 5

Decode_Frame_Timewin = pd.DataFrame(columns=['Train_Shuffle','Test_Shuffle','Train_Window','Test_Window','Accuracy'])


#%%
meta = metamer_ids.copy().reset_index(drop=True)
meta['trial_row'] = np.arange(len(meta), dtype=int)
meta = meta.loc[meta['Img_Index'].isin(img_range)].copy()

trial_indices = {}
for (img, shuf), grp in meta.groupby(['Img_Index', 'Shuffle_Level'], sort=False):
    idx = grp['trial_row'].to_numpy(dtype=int)
    trial_indices[(int(img), int(shuf))] = np.sort(idx)
    assert len(idx) == 5, (img, shuf, len(idx))


def collect_indices_by_shuffle(trial_map, img_ids, shuffle_level):
    idxs = []
    labels = []
    for img in img_ids:
        ids = trial_map[(int(img), int(shuffle_level))]
        idxs.extend(ids.tolist())
        labels.extend([int(img)] * len(ids))
    return np.asarray(idxs, dtype=int), np.asarray(labels, dtype=int)


def build_window_feature(resp_pdf, trial_rows, win_start, win_len):
    """Return X in shape (N_Selection, N_Cell)."""
    trial_slice = resp_pdf[:, trial_rows, win_start : win_start + win_len]
    return trial_slice.sum(axis=-1).T


win_starts = list(range(0, n_t - win_size + 1, win_step))
def run_decode_for_group(resp_pdf, group_name):
    decoder_rows = []
    n_cells_group = int(resp_pdf.shape[0])
    if n_cells_group == 0:
        return decoder_rows
    for train_shuffle in range(5):
        train_idx, y_train = collect_indices_by_shuffle(trial_indices, img_range, train_shuffle)
        if np.unique(y_train).size != len(img_range):
            raise ValueError(f'Train shuffle {train_shuffle} does not cover all classes.')

        for train_win in tqdm(win_starts, desc=f'{group_name} train window'):
            X_train = build_window_feature(resp_pdf, train_idx, train_win, win_size)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel='linear', decision_function_shape='ovr'),
            )
            cv_acc = float(cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy').mean())
            clf.fit(X_train, y_train)

            for test_shuffle in range(5):
                test_idx, y_test = collect_indices_by_shuffle(trial_indices, img_range, test_shuffle)
                for test_win in win_starts:
                    # Avoid leakage when train/test use same shuffle and overlapping time bins.
                    # Instead of backfilling with cv_acc, mark as NaN to avoid artificial diagonal bands.
                    same_shuffle = int(train_shuffle) == int(test_shuffle)
                    overlap_in_time = abs(int(train_win) - int(test_win)) < int(win_size)
                    if same_shuffle and overlap_in_time:
                        acc = np.nan
                        n_test_now = 0
                    else:
                        X_test = build_window_feature(resp_pdf, test_idx, test_win, win_size)
                        acc = float(clf.score(X_test, y_test))
                        n_test_now = int(X_test.shape[0])
                    decoder_rows.append(
                        {
                            'Cell_Group': str(group_name),
                            'Train_Shuffle': int(train_shuffle),
                            'Test_Shuffle': int(test_shuffle),
                            'Train_Window': int(train_win),
                            'Test_Window': int(test_win),
                            'Accuracy': acc,
                            'CV_Accuracy': cv_acc,
                            'n_cells_group': n_cells_group,
                            'n_train': int(X_train.shape[0]),
                            'n_test': n_test_now,
                            'win_size': int(win_size),
                            'win_step': int(win_step),
                            'Brain_Area': used_ares,
                        }
                    )
    return decoder_rows


all_decoder_rows = []
all_decoder_rows.extend(run_decode_for_group(psth_pdf, 'all'))
for cat in np.sort(np.unique(cell_cats)):
    cat_mask = cell_cats == cat
    cat_resp = psth_pdf[cat_mask, :, :]
    all_decoder_rows.extend(run_decode_for_group(cat_resp, f'cat_{int(cat)}'))

Decode_Frame_Timewin = pd.DataFrame(all_decoder_rows)

save_name = f'{used_ares}_decode_svm_timewin_img20_win{win_size}_step{win_step}'
Decode_Frame_Timewin.to_parquet(ot.Join(wp, f'{save_name}.parquet'), index=False)
# Decode_Frame_Timewin.to_csv(ot.Join(wp, f'{save_name}.csv'), index=False)
#%%


train_shuffle = 4
test_shuffle = 0
x_ms_min, x_ms_max = 0, 250
y_ms_min, y_ms_max = 0, 250
vmax=1
# Use one group name (e.g. ['cat_1']) or many (e.g. ['all', 'cat_0', 'cat_1', 'cat_2'])
# plot_cell_groups = ['all', 'cat_0', 'cat_1', 'cat_2']
plot_cell_groups = ['all']

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

