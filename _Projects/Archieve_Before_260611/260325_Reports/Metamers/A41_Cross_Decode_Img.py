'''
This script is to decode cross different metamer constrain level.

'''

#%%
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

wp = r'E:\#Preprocessed_Data\260305_Report_Data\Decoding_SVM_MetamerOnly'
datafoler = r'E:\#Preprocessed_Data\Selected_Cells'
brain_sites = ['AL','ASB','ML','MSB']
avr_resp = {}

# load average response for each site.
for site in tqdm(brain_sites):
    data = np.load(ot.Join(datafoler,f'{site}_Cells_Metamer_Only.npz'),allow_pickle=True)
    c_rsp = data['psth'][:,:,160:320].sum(-1)
    avr_resp[site] = c_rsp

#%% generate data of different constrain level.
# avr_resp[site] shape: (N_Cell, 1000)
# Trial sequence (1-based): indices 1-200 = repeat 1, 201-400 = repeat 2, ..., 801-1000 = repeat 5.
# Within each 200: 1-40 = shuffle0, 41-80 = shuffle1, 81-120 = shuffle2, 121-160 = shuffle3, 161-200 = shuffle4.
# So flat_index (0-based) = repeat*200 + (shuffle*40 + img_index).

N_REPEAT = 5
N_SHUFFLE = 5
N_IMG = 40
N_COND = N_SHUFFLE * N_IMG   # 200
N_TRIAL = N_REPEAT * N_COND  # 1000

def build_by_shuffle(r: np.ndarray) -> dict:
    """
    r: (n_cell, 1000). Returns dict[shuffle_level] -> (n_cell, 40, 5)
    with axis 1 = img_index (0..39), axis 2 = repeat (0..4).
    Assumes: indices 0-199 = repeat 0, 200-399 = repeat 1, ...; within 200, 0-39 = shuffle0, 40-79 = shuffle1, ...
    """
    n_cell = r.shape[0]
    # (n_cell, 5, 200): dim1 = repeat (0..4), dim2 = cond within repeat (0-39 sh0, 40-79 sh1, ...)
    r2 = r.reshape(n_cell, N_REPEAT, N_COND)
    out = {}
    for s in range(N_SHUFFLE):
        # within each repeat, conditions s*40 .. (s+1)*40 are this shuffle's 40 images
        out[s] = r2[:, :, s * N_IMG : (s + 1) * N_IMG].transpose(0, 2, 1)
        # (n_cell, 40, 5): axis1=img_index (0..39), axis2=repeat (0..4)
    return out

def trial_index(shuffle: int, img_index: int, repeat: int) -> int:
    """Flat trial index (0-based) in 1000: repeat first, then within 200: shuffle0(40), shuffle1(40), ..., img_index within shuffle."""
    cond_200 = shuffle * N_IMG + img_index
    return repeat * N_COND + cond_200

# Per-site structure: extract data by shuffle level, keep img index.
# resp_by_site_shuffle[site][shuffle] -> (n_cell, 40, 5); dim1=img_index, dim2=repeat
resp_by_site_shuffle = {}
for site in brain_sites:
    resp_by_site_shuffle[site] = build_by_shuffle(avr_resp[site])

# Optional: for each site, DataFrame of (shuffle, img_index, repeat) -> trial index
stim_index = []
for s in range(N_SHUFFLE):
    for img_i in range(N_IMG):
        for rep in range(N_REPEAT):
            stim_index.append({
                'Shuffle': s,
                'Img_Index': img_i,
                'Repeat': rep,
                'Trial_Index': trial_index(s, img_i, rep),
                'Cond_200': s * N_IMG + img_i,
            })
stim_lookup = pd.DataFrame(stim_index)
stim_lookup['Ani'] = stim_lookup['Img_Index'] <20

#%%
######## SVM builder ########
'''
This part is to build SVM classifier to test decoding ability cross different metamer constrain level.

train SVM use linear kernel, 5 fold, and predict image index. It's a category classification problem.


'''

def get_X_y_for_ani(resp: np.ndarray, ani: int) -> tuple:
    """
    resp: (n_cell, 40, 5). ani in {0, 1}: 0 = inanimate (img 20-39), 1 = animate (img 0-19).
    Returns X (n_samples, n_cell), y (n_samples,) with class labels 0..19.
    """
    if ani == 1:
        img_slice = slice(0, 20)
    else:
        img_slice = slice(20, 40)
    # (n_cell, 20, 5) -> flatten to (20*5, n_cell) = (100, n_cell)
    r = resp[:, img_slice, :]   # (n_cell, 20, 5)
    n_cell = r.shape[0]
    X = r.transpose(1, 2, 0).reshape(-1, n_cell)   # (100, n_cell)
    y = np.repeat(np.arange(20), 5)                 # (100,)
    return X, y

N_FOLD = 5
rows_decoding = []
for area in tqdm(brain_sites, desc='Area'):
    for ani in [0, 1]:
        for train_sh in range(N_SHUFFLE):
            for test_sh in range(N_SHUFFLE):
                X_train, y_train = get_X_y_for_ani(resp_by_site_shuffle[area][train_sh], ani)
                X_test, y_test = get_X_y_for_ani(resp_by_site_shuffle[area][test_sh], ani)
                if train_sh == test_sh:
                    # 5-fold CV: leave one repeat out (each repeat has all 20 classes)
                    # Row i = img i//5, repeat i%5; group by repeat so test fold has all classes
                    groups_rep = np.arange(len(y_train)) % N_FOLD  # 0,1,2,3,4,0,1,2,3,4,...
                    kf = GroupKFold(n_splits=N_FOLD)
                    clf = SVC(kernel='linear')
                    y_pred = cross_val_predict(clf, X_train, y_train, groups=groups_rep, cv=kf)
                    correct_rate = (y_pred == y_train).mean()
                else:
                    clf = SVC(kernel='linear').fit(X_train, y_train)
                    correct_rate = (clf.predict(X_test) == y_test).mean()
                rows_decoding.append({
                    'Area': area,
                    'Ani': ani,
                    'Train_Shuffle': train_sh,
                    'Test_Shuffle': test_sh,
                    'Correct_Rate': float(correct_rate),
                })

decoding_cr = pd.DataFrame(rows_decoding, columns=['Area', 'Ani', 'Train_Shuffle', 'Test_Shuffle', 'Correct_Rate'])

#%% Heatmaps: decoding correct rate across shuffle levels (Animate left, Inanimate right)
# Select brain area(s) to include; use mean over selected areas for the 5x5 matrix
heatmap_areas = ['AL']  # adjustable: e.g. ['AL'] or ['AL', 'MSB']

df_hm = decoding_cr.loc[decoding_cr['Area'].isin(heatmap_areas)]
pivot_ani = (
    df_hm.loc[df_hm['Ani'] == 1]
    .groupby(['Train_Shuffle', 'Test_Shuffle'], as_index=False)['Correct_Rate']
    .mean()
)
pivot_inani = (
    df_hm.loc[df_hm['Ani'] == 0]
    .groupby(['Train_Shuffle', 'Test_Shuffle'], as_index=False)['Correct_Rate']
    .mean()
)

def to_matrix(pivot_df):
    m = pivot_df.pivot(index='Train_Shuffle', columns='Test_Shuffle', values='Correct_Rate')
    m = m.reindex(index=range(5), columns=range(5))
    return m

mat_ani = to_matrix(pivot_ani)
mat_inani = to_matrix(pivot_inani)

fig_hm, axes_hm = plt.subplots(1, 2, figsize=(10, 4.5), dpi=240,sharey=True)
sns.heatmap(
    mat_ani,
    ax=axes_hm[0],
    annot=True,
    fmt='.2f',
    cmap='viridis',
    vmin=0,
    vmax=1,
    xticklabels=['0', '1', '2', '3', '4'],
    yticklabels=['0', '1', '2', '3', '4'],
    cbar=False,square=True,
)
axes_hm[0].set_xlabel('Test Shuffle')
axes_hm[0].set_ylabel('Train Shuffle')
axes_hm[0].set_title(f'Animate ({", ".join(heatmap_areas)})')

sns.heatmap(
    mat_inani,
    ax=axes_hm[1],
    annot=True,
    fmt='.2f',
    cmap='viridis',
    vmin=0,
    vmax=1,
    xticklabels=['0', '1', '2', '3', '4'],
    yticklabels=['0', '1', '2', '3', '4'],
    cbar_kws={'label': 'Correct rate'},square=True,
)
# axes_hm[1].set_xlabel('Test Shuffle')
axes_hm[1].set_ylabel('')
axes_hm[1].set_title(f'Inanimate ({", ".join(heatmap_areas)})')

plt.tight_layout()
plt.show()


#%%
####################### Part2, decode shuffle level ########

'''
Similar to the image-identity decoding above, here we decode current
constrain level (shuffle 0-4) for each image, separately for animate
and inanimate. We treat shuffle level as a linear variable (0..4) and
use SVR with 5-fold cross-validation over repeats. For reporting we
round the predicted level back to {0..4} and compute a \"correct rate\".
'''

def get_X_y_shuffle(area: str, img_idx: int) -> tuple:
    """
    Build X, y for decoding shuffle level of one image in one area.
    X: (25, n_cell), y: (25,) with real-valued targets 0..4 (shuffle levels).
    """
    # Collect responses: for each shuffle, 5 repeats
    mats = []
    labels = []
    for s in range(N_SHUFFLE):
        r = resp_by_site_shuffle[area][s][:, img_idx, :]  # (n_cell, 5)
        n_cell = r.shape[0]
        mats.append(r.T)                  # (5, n_cell)
        labels.append(np.repeat(float(s), 5))    # (5,)
    X = np.vstack(mats)                   # (25, n_cell)
    y = np.concatenate(labels)            # (25,)
    return X, y


rows_shuffle = []
for area in tqdm(brain_sites, desc='Area_shuffle'):
    for ani in [0, 1]:
        if ani == 1:
            img_range = range(0, 20)
        else:
            img_range = range(20, 40)
        for img_idx in img_range:
            X, y = get_X_y_shuffle(area, img_idx)
            # Group by repeat index across shuffles: 25 samples = 5 repeats × 5 shuffles
            groups_rep = np.tile(np.arange(N_REPEAT), N_SHUFFLE)  # 0..4 repeated per shuffle
            gkf = GroupKFold(n_splits=N_FOLD)
            # SVR: treat shuffle level as linear / continuous 0..4
            from sklearn.svm import SVR
            svr = SVR(kernel='linear')
            y_pred = cross_val_predict(svr, X, y, groups=groups_rep, cv=gkf)
            # Discretize predictions back to levels 0..4 for a \"correct rate\"
            y_pred_disc = np.rint(y_pred).clip(0, N_SHUFFLE - 1).astype(int)
            cr = (y_pred_disc == y.astype(int)).mean()
            rows_shuffle.append(
                {
                    'Area': area,
                    'Ani': ani,
                    'Img_Index': int(img_idx),
                    'Correct_Rate': float(cr),
                }
            )

decoding_shuffle_cr = pd.DataFrame(
    rows_shuffle,
    columns=['Area', 'Ani', 'Img_Index', 'Correct_Rate'],
)
#%%

# Map Ani values to descriptive labels for legend
decoding_shuffle_cr_plot = decoding_shuffle_cr.copy()
ani_map = {0: 'Inanimate', 1: 'Animate'}
decoding_shuffle_cr_plot['Ani_Label'] = decoding_shuffle_cr_plot['Ani'].map(ani_map)

# Use tab10 colors: Animate as orange (tab:orange), Inanimate as blue (tab:blue)
tab10_colors = plt.get_cmap('tab10')
palette = {'Inanimate': tab10_colors(0), 'Animate': tab10_colors(1)}  # 0: blue, 1: orange

fig, ax = plt.subplots(ncols=1, nrows=1, dpi=240, figsize=(7, 5))

sns.boxplot(
    x='Area',
    y='Correct_Rate',
    data=decoding_shuffle_cr_plot,
    hue='Ani_Label',
    ax=ax,
    palette=palette,
    whis=[5,95],
    showfliers=False,
    linewidth=2,
    width=0.55,
    boxprops=dict(alpha=0.90),
)

# Add reference line and beautify
ax.axhline(0.2, linestyle='--', color='gray', alpha=0.5, lw=1.3)
ax.set_xticklabels(brain_sites, rotation=20, ha='right')
ax.set_ylabel('Correct Rate', fontsize=13, labelpad=5)
ax.set_xlabel('Brain Area', fontsize=13, labelpad=5)
ax.set_title('Decoding Shuffle Level', fontsize=15, fontweight='bold', pad=12)
ax.set_ylim(0, 1.05)

# Beautify legend
h, l = ax.get_legend_handles_labels()
ax.legend(
    h, l,
    title="Category",
    title_fontsize=12,
    fontsize=11,
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="0.85"
)

sns.despine(ax=ax)
plt.tight_layout()
plt.show()

#%%

from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, cross_val_predict

#%%
####################### Part3, decode shuffle level (ignore Img_Index) ########

'''
Here we decode current constrain level (shuffle 0-4) for each Area and Ani,
but we ignore image identity: all frames (images) within an Ani condition
are pooled together. Shuffle level is treated as a linear variable (0..4),
decoded with SVR + 5-fold GroupKFold over repeats. For reporting we round
predicted levels back to {0..4} and compute a "correct rate".
'''

def get_X_y_shuffle_pooled(area: str, ani: int) -> tuple:
    """
    Build X, y for decoding shuffle level in one area and one Ani group,
    pooling over all images in that Ani group.

    X: (N_samples, n_cell), y: (N_samples,) with real-valued targets 0..4.
       N_samples = N_SHUFFLE * n_img_ani * N_REPEAT.
    """
    if ani == 1:
        img_range = range(0, 20)   # Animate images
    else:
        img_range = range(20, 40)  # Inanimate images

    mats = []
    labels = []
    # For grouping by repeat across all shuffles and images:
    # we will build groups in the same loop.
    groups = []

    for s in range(N_SHUFFLE):
        # resp_by_site_shuffle[area][s]: (n_cell, 40, 5)
        r = resp_by_site_shuffle[area][s][:, img_range, :]     # (n_cell, n_img_ani, 5)
        n_cell = r.shape[0]
        n_img_ani = len(img_range)

        # (n_cell, n_img_ani, 5) -> (n_img_ani, 5, n_cell) -> (n_img_ani*5, n_cell)
        r_reshaped = r.transpose(1, 2, 0).reshape(-1, n_cell)  # (n_img_ani*5, n_cell)
        mats.append(r_reshaped)

        # Labels: shuffle level s for each (img, repeat) sample
        labels.append(np.repeat(float(s), n_img_ani * N_REPEAT))

        # Groups: repeat index 0..4 for each image, repeated per shuffle
        # For each shuffle, pattern per image is [0,1,2,3,4]; stacked over images.
        rep_pattern = np.tile(np.arange(N_REPEAT), n_img_ani)  # length = n_img_ani*5
        groups.append(rep_pattern)

    X = np.vstack(mats)                                     # (N_SHUFFLE*n_img_ani*5, n_cell)
    y = np.concatenate(labels)                              # (N_SHUFFLE*n_img_ani*5,)
    groups = np.concatenate(groups)                         # same length as y

    return X, y, groups


rows_shuffle_pooled = []
for area in tqdm(brain_sites, desc='Area_shuffle_pooled'):
    for ani in [0, 1]:
        X, y, groups_rep = get_X_y_shuffle_pooled(area, ani)
        gkf = GroupKFold(n_splits=N_FOLD)
        svr = SVR(kernel='linear')

        # SVR regression on shuffle level (0..4)
        y_pred = cross_val_predict(svr, X, y, groups=groups_rep, cv=gkf)

        # Discretize predictions to levels 0..4 for an accuracy-like metric
        y_pred_disc = np.rint(y_pred).clip(0, N_SHUFFLE - 1).astype(int)
        cr = (y_pred_disc == y.astype(int)).mean()

        rows_shuffle_pooled.append(
            {
                'Area': area,
                'Ani': ani,
                'Correct_Rate': float(cr),
            }
        )

decoding_shuffle_pooled_cr = pd.DataFrame(
    rows_shuffle_pooled,
    columns=['Area', 'Ani', 'Correct_Rate'],
)

#%% Bootstrap control for pooled shuffle decoding (label shuffling, 10 repeats)

rows_shuffle_pooled_boot = []
N_BOOT = 100

for area in brain_sites:
    for ani in [0, 1]:
        X, y, groups_rep = get_X_y_shuffle_pooled(area, ani)
        gkf = GroupKFold(n_splits=N_FOLD)
        for b in tqdm(range(N_BOOT)):
            # Randomly permute shuffle labels (y) while keeping X and groups fixed
            y_shuffled = np.random.permutation(y)
            svr = SVR(kernel='linear')
            y_pred = cross_val_predict(svr, X, y_shuffled, groups=groups_rep, cv=gkf)
            y_pred_disc = np.rint(y_pred).clip(0, N_SHUFFLE - 1).astype(int)
            cr_boot = (y_pred_disc == y_shuffled.astype(int)).mean()
            rows_shuffle_pooled_boot.append(
                {
                    'Area': area,
                    'Ani': ani,
                    'Bootstrap': b,
                    'Correct_Rate': float(cr_boot),
                }
            )

decoding_shuffle_pooled_boot = pd.DataFrame(
    rows_shuffle_pooled_boot,
    columns=['Area', 'Ani', 'Bootstrap', 'Correct_Rate'],
)

#%% ####################### Part3, Cross Img id decode ########

'''
Cross-image decoding of shuffle level (0-4), ignoring animate/inanimate.
For each Area, we compute a 40x40 matrix (Train_Img x Test_Img) of
SVR-based decoding performance:

- For train_img == test_img: 5-fold GroupKFold over repeats (as before).
- For train_img != test_img: train on all 25 trials of train_img,
  test on all 25 trials of test_img.

Performance metric: round predictions to 0..4 and take fraction correct (same as pooled decoding).
'''

rows_img_cross = []
for area in tqdm(brain_sites, desc='Area_shuffle_cross_img'):
    for train_img in range(N_IMG):
        X_train, y_train = get_X_y_shuffle(area, train_img)
        n_samples = len(y_train)
        for test_img in range(N_IMG):
            X_test, y_test = get_X_y_shuffle(area, test_img)
            if train_img == test_img:
                groups_rep = np.arange(n_samples) % N_REPEAT
                gkf = GroupKFold(n_splits=N_FOLD)
                svr = SVR(kernel='linear')
                y_pred = cross_val_predict(svr, X_train, y_train, groups=groups_rep, cv=gkf)
                y_true = y_train
            else:
                svr = SVR(kernel='linear').fit(X_train, y_train)
                y_pred = svr.predict(X_test)
                y_true = y_test

            # Same metric as pooled decoding: discretize predictions, then fraction correct
            y_pred_disc = np.rint(y_pred).clip(0, N_SHUFFLE - 1).astype(int)
            cr = float((y_pred_disc == y_true.astype(int)).mean())
            rows_img_cross.append(
                {
                    'Area': area,
                    'Train_Img': int(train_img),
                    'Test_Img': int(test_img),
                    'Correct_Rate': cr,
                }
            )

decoding_shuffle_img_cross_cr = pd.DataFrame(
    rows_img_cross,
    columns=['Area', 'Train_Img', 'Test_Img', 'Correct_Rate'],
)

#%% Heatmaps of cross-image shuffle decoding (2x2 subplots for 4 areas)

area_order = ['MSB', 'ML', 'ASB', 'AL']  # adjust order if needed

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axes = plt.subplots(2, 2, figsize=(9, 7), dpi=240, sharex=False, sharey=False)
axes = axes.ravel()

vmin, vmax = 0.0, 0.4
im = None

for ax, area in zip(axes, area_order):
    df_area = decoding_shuffle_img_cross_cr.loc[decoding_shuffle_img_cross_cr['Area'] == area]
    mat = df_area.pivot(index='Train_Img', columns='Test_Img', values='Correct_Rate')
    # Ensure full 0..39 coverage even if some rows/cols are missing
    mat = mat.reindex(index=range(N_IMG), columns=range(N_IMG))

    im = sns.heatmap(
        mat,
        ax=ax,
        center=0.2,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        square=True,
    )

    # Annotate Ani (0–19) and Inani (20–39) on both axes with 2 labels
    ax.set_xticks([9.5, 29.5])
    ax.set_xticklabels(['Ani', 'Inani'], rotation=0)
    ax.set_yticks([9.5, 29.5])
    ax.set_yticklabels(['Ani', 'Inani'], rotation=90)
    ax.plot([0, 40], [20, 20], linestyle='--', color='yellow', zorder=9)
    ax.plot([20, 20], [0, 40], linestyle='--', color='yellow', zorder=9)
    ax.set_title(area)

# Only outer labels: bottom row x-labels, top row y-labels
for i, ax in enumerate(axes):
    row, col = divmod(i, 2)
    # x-labels only on bottom row (row == 1)
    if row == 1:
        ax.set_xlabel('Test image group')
    else:
        ax.set_xlabel('')
    # y-labels only on top row (row == 0)
    if col == 0:
        ax.set_ylabel('Train image group')
    else:
        ax.set_ylabel('')

# Shared colorbar on the right (use all axes so sizes stay matched)
if im is not None:
    cbar = fig.colorbar(im.collections[0], ax=axes, location='right', fraction=0.1, pad=0.04)
    cbar.set_label('Correct Rate')

# plt.tight_layout()
plt.show()
#%%
#%% Boxplot: cross-image decoding by Ani/Inani train/test pairing

df_box = decoding_shuffle_img_cross_cr.copy()
df_box = df_box[df_box.Test_Img != df_box.Train_Img]
# Define Ani/Inani for train and test images
df_box['Train_Ani'] = df_box['Train_Img'] < 20
df_box['Test_Ani'] = df_box['Test_Img'] < 20

def _pair_label(row):
    if row['Train_Ani'] and row['Test_Ani']:
        return 'Ani → Ani'
    if (not row['Train_Ani']) and (not row['Test_Ani']):
        return 'Inani → Inani'
    if row['Train_Ani'] and (not row['Test_Ani']):
        return 'Ani → Inani'
    if (not row['Train_Ani']) and row['Test_Ani']:
        return 'Inani → Ani'
    return None

df_box['Pair_Label'] = df_box.apply(_pair_label, axis=1)

# Desired x order: ['Ani → Ani', 'Inani → Inani', 'Ani → Inani', 'Inani → Ani']
x_order = ['Ani → Ani', 'Inani → Inani', 'Ani → Inani', 'Inani → Ani']

palette_pairs = {
    'MSB': 'tab:orange',
    'ML': '#ffcc8a',
    'ASB': '#8ab5ff',
    'AL': 'tab:blue',
}

fig_pair, ax_pair = plt.subplots(figsize=(6, 5), dpi=240)
sns.boxplot(
    data=df_box,
    x='Pair_Label',
    y='Correct_Rate',
    hue='Area',
    order=x_order,
    palette=palette_pairs,
    whis=(5, 95),
    showfliers=False,
    ax=ax_pair,width=0.55,
)
ax_pair.set_xlabel('Train → Test',fontsize=15, fontweight='bold')
ax_pair.set_ylabel('Correct Rate',fontsize=15, fontweight='bold')
ax_pair.set_title('Cross-image decoding')
ax_pair.legend(title='Condition', fontsize=8,title_fontsize=10)
ax_pair.axhline(0.2,linestyle='--',color='gray',alpha=0.5,lw=1.3)
plt.tight_layout()
plt.show()


