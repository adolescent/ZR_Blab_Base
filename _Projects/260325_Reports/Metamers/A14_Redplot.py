'''
This script will generate redplots for all cell.

Including FOB redplot and 
'''

#%%


from nt import read
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


def sort_by_pca1(x, reverse=True):
    """
    x: array of shape (n_cell, n_dim)
    Returns x sorted by scores on the first principal component.
    If reverse is True, sort in descending order; otherwise ascending.
    """
    x = np.asarray(x)
    # center along feature dimension
    x_centered = x - x.mean(axis=0, keepdims=True)
    # covariance matrix of features
    cov = np.cov(x_centered, rowvar=False)
    # eigh since covariance is symmetric
    eigvals, eigvecs = np.linalg.eigh(cov)
    # first PC -> eigenvector with largest eigenvalue
    pc1 = eigvecs[:, np.argmax(eigvals)]
    # scores of each cell on PC1
    scores = x_centered @ pc1
    order = np.argsort(scores)
    if reverse:
        order = order[::-1]
    return x[order], order, scores[order]


datafolder=r'E:\#Preprocessed_Data\Selected_Cells'
savepath = r'E:\#Preprocessed_Data\260305_Report_Data\Site_ANOVAs'
brain_area = 'AL'
file = f'{brain_area}_Cells_Metamer_Only.npz'

reader = np.load(ot.Join(datafolder,file),allow_pickle=True)
psth = reader['psth'][:,:,160:320].sum(-1)
if brain_area == 'MSB':
    fob = reader['FOB'][160:,:,160:320].sum(-1)
else:
    fob = reader['FOB'][:,:,160:320].sum(-1)
# cell 158 start are sti 150.
# del reader
#%% ############## Part1, plot FOB infos. ###################
plotable = fob.reshape((len(fob)), 4, 75).mean(1)
plotable = plotable / plotable.std(1, keepdims=True)
plotable, plot_order, plot_scores = sort_by_pca1(plotable, reverse=True)

# Custom y and x ticks
y_ticks = np.arange(0,3500,200)
y_ticklabels = [str(v) for v in y_ticks]
x_annot_labels = ['Face', 'Body', 'Object', 'Scene', 'Food']

# Prepare x-axis category labels: there are 75 columns, 15 for each category
x_locs = np.arange(0, 75, 15) + 7  # Label at the center of each block of 15
x_label_list = []
for i in range(75):
    if i in x_locs:
        idx = list(x_locs).index(i)
        x_label_list.append(x_annot_labels[idx])
    else:
        x_label_list.append('')

fig = plt.figure(figsize=(8, 6), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.2)

ax = fig.add_subplot(gs[0, 0])

# Create a cax that is only the central part of the right side
# [left, bottom, width, height] in figure fraction
cax = fig.add_axes([0.83, 0.25, 0.025, 0.5])  # adjust as needed for half height

hm = sns.heatmap(
    plotable,
    cmap='bwr',
    center=0,
    vmax=6,
    ax=ax,
    cbar_ax=cax,
    cbar_kws={'shrink': 1}  # shrink doesn't do anything with cbar_ax, so safe to leave or remove
)

# y-ticks: select only valid indices (plotable.shape[0] may be < max(y_ticks))
yticks_valid = [yt for yt in y_ticks if yt < plotable.shape[0]]
ax.set_yticks(yticks_valid)
ax.set_yticklabels([str(yt) for yt in yticks_valid], fontsize=12)

# x-ticks: annotate only at the midpoint of each 15-block
ax.set_xticks(x_locs)
ax.set_xticklabels(x_annot_labels, fontsize=13, rotation=30, ha='center')

# Remove small x-tickmarks and add major grid for visual separation
ax.tick_params(axis='x', length=0)
ax.set_xlabel("Category", fontsize=15)
ax.set_ylabel(f"{brain_area} Cells", fontsize=15)

# ax.set_title('Normalized Average Response to FOB Categories', fontsize=16, pad=20)
plt.tight_layout()
plt.show()


#%% redplot of constrained response.
metamer_rsp = psth.reshape((len(psth),5,200)).mean(1)
plotable = metamer_rsp[plot_order,:]
plotable= plotable / plotable.std(1, keepdims=True)


y_ticks = np.arange(0,3500,200)
y_ticklabels = [str(v) for v in y_ticks]
x_annot_labels = ['Raw', 'C4', 'C3', 'C2', 'C1']

# Prepare x-axis category labels: there are 75 columns, 15 for each category
x_locs = np.arange(0, 200, 40) + 20  # Label at the center of each block of 15
x_label_list = []
for i in range(75):
    if i in x_locs:
        idx = list(x_locs).index(i)
        x_label_list.append(x_annot_labels[idx])
    else:
        x_label_list.append('')

fig = plt.figure(figsize=(12, 6), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.2)

ax = fig.add_subplot(gs[0, 0])

# Create a cax that is only the central part of the right side
# [left, bottom, width, height] in figure fraction
cax = fig.add_axes([0.83, 0.25, 0.025, 0.5])  # adjust as needed for half height

hm = sns.heatmap(
    plotable,
    cmap='bwr',
    center=0,
    vmax=6,
    ax=ax,
    cbar_ax=cax,
    cbar_kws={'shrink': 1}  # shrink doesn't do anything with cbar_ax, so safe to leave or remove
)

# Add yellow vertical lines every 40 columns (at col-edges), i.e. after 40, 80, 120, 160
for vline in [40, 80, 120, 160]:
    ax.axvline(vline, color='blue',alpha=0.5, linewidth=2)

# y-ticks: select only valid indices (plotable.shape[0] may be < max(y_ticks))
yticks_valid = [yt for yt in y_ticks if yt < plotable.shape[0]]
ax.set_yticks(yticks_valid)
ax.set_yticklabels([str(yt) for yt in yticks_valid], fontsize=12)

# x-ticks: annotate only at the midpoint of each 15-block
ax.set_xticks(x_locs)
ax.set_xticklabels(x_annot_labels, fontsize=13, rotation=30, ha='center')

# Remove small x-tickmarks and add major grid for visual separation
ax.tick_params(axis='x', length=0)
ax.set_xlabel("Category", fontsize=15)
ax.set_ylabel(f"{brain_area} Cells", fontsize=15)

# ax.set_title('Normalized Average Response to FOB Categories', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

#%%
