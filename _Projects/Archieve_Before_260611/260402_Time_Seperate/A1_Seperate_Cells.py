"""
Simple neuron clustering pipeline for Jupyter use.

Steps:
1) Average PSTH over images -> (N_cell, N_time)
2) Z-score each cell response in 0-250 ms
3) PCA on z-scored responses (number of PCs adjustable)
4) KMeans on PCA results
5) Plot class-averaged responses and save each cell class
"""

#%%
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#%% User settings
wp = r"E:\#Preprocessed_Data\Selected_Cells"
brain_area = 'ML'
npz_name = f"{brain_area}_Cells_Metamer_Only.npz"
savepath=r'E:\#Preprocessed_Data\260402_TC_Analysis'
win_start_ms = 25
win_end_ms = 250
n_clusters = 3
var_keep = 0.80  # adjustable explained variance target (0-1]

#%% Load PSTH
msb_resps = np.load(os.path.join(wp, npz_name))
raw_psth = msb_resps["psth"]  # expected: (N_cell, N_img, N_time)

assert raw_psth.ndim == 3, f"Expected 3D psth, got {raw_psth.shape}"
n_cells, n_imgs, n_t = raw_psth.shape
print(f"Loaded psth shape: cells={n_cells}, imgs={n_imgs}, time_bins={n_t}")

# Build time axis from -100 ms with 1 ms bins
time_ms = np.arange(-100, -100 + n_t)
pdf_norm_mask = (time_ms >= 0) & (time_ms <= 300)  # fixed normalization range
pdf_time_axis = time_ms[pdf_norm_mask]
if pdf_time_axis.size == 0:
    raise ValueError("No bins in fixed PDF normalization window 0-300 ms.")

#%% 1) Average PSTH over images -> (N_cell, N_time)
cell_time = raw_psth.mean(axis=1)
print(f"cell_time shape: {cell_time.shape}")

#%% 2) Convert to per-cell PDF, always normalized in 0-300 ms
cell_pdf_win = cell_time[:, pdf_norm_mask]

# Shift each cell to non-negative before normalizing to probability density.
cell_win_shift = cell_pdf_win - cell_pdf_win.min(axis=1, keepdims=True)
sum_per_cell = cell_win_shift.sum(axis=1, keepdims=True)
sum_per_cell = np.where(sum_per_cell <= 1e-12, 1.0, sum_per_cell)
pdf_time_all = cell_win_shift / sum_per_cell

# Optional analysis sub-window after PDF normalization.
analysis_mask = (pdf_time_axis >= win_start_ms) & (pdf_time_axis <= win_end_ms)
time_win = pdf_time_axis[analysis_mask]
if time_win.size == 0:
    raise ValueError(
        f"No bins in analysis window {win_start_ms}-{win_end_ms} ms "
        "within fixed PDF range 0-300 ms."
    )
pdf_time = pdf_time_all[:, analysis_mask]
print(f"pdf_time shape: {pdf_time.shape}")

#%% 3) PCA on PDF time responses (auto PC selection by explained variance)
if not (0 < var_keep <= 1):
    raise ValueError(f"var_keep must be in (0, 1], got {var_keep}")

pca = PCA(n_components=var_keep, random_state=0)
pc_scores = pca.fit_transform(pdf_time)
n_pcs = pca.n_components_
print(f"PCA score shape: {pc_scores.shape}")
print(f"Selected n_pcs: {n_pcs} for var_keep={var_keep:.2f}")
print(f"Explained variance ratio (first 5 PCs): {pca.explained_variance_ratio_[:5]}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")
#%% 4) KMeans on PCA results
kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
labels = kmeans.fit_predict(pc_scores)
print("Cluster counts:", np.bincount(labels))

#%% 5) Save labels/features and make plots
out_dir = savepath

out_npz = os.path.join(out_dir, f"{brain_area}_pca_kmeans3_ztime_0to{win_end_ms}ms.npz")
np.savez_compressed(
    out_npz,
    labels=labels,
    pc_scores=pc_scores,
    explained_variance_ratio=pca.explained_variance_ratio_,
    pdf_time_all=pdf_time_all,
    pdf_time=pdf_time,
    cell_time=cell_time,
    time_ms=time_ms,
    pdf_norm_mask=pdf_norm_mask,
    analysis_mask=analysis_mask,
)
print(f"Saved: {out_npz}")

# PCA scatter (PC1 vs PC2)
palette = plt.get_cmap("tab10").colors[:n_clusters]
plt.figure(figsize=(7.5, 6.5))
for k in range(n_clusters):
    idx = labels == k
    plt.scatter(pc_scores[idx, 0], pc_scores[idx, 1], s=16, alpha=0.85, color=palette[k], label=f"class {k} (n={idx.sum()})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Neuron PCA (PDF norm 0-300ms, analyze {win_start_ms}-{win_end_ms}ms, keep {n_pcs} PCs)")
plt.legend(frameon=False, loc="best")
plt.tight_layout()

out_png1 = os.path.join(out_dir, f"{brain_area}_pca_kmeans3_ztime_0to{win_end_ms}ms.png")
plt.savefig(out_png1, dpi=200)
print(f"Saved: {out_png1}")
plt.show()

# Class-averaged PDF curves over full 0-300 ms (mean +/- SEM across cells)
plt.figure(figsize=(8.5, 5.0))
for k in range(n_clusters):
    idx = labels == k
    if idx.sum() == 0:
        continue
    class_curves = pdf_time_all[idx, :]
    mean_curve = class_curves.mean(axis=0)
    sem_curve = class_curves.std(axis=0, ddof=1) / np.sqrt(class_curves.shape[0])
    plt.plot(pdf_time_axis, mean_curve, lw=2.0, color=palette[k], label=f"class {k} (n={idx.sum()})")
    plt.fill_between(pdf_time_axis, mean_curve - sem_curve, mean_curve + sem_curve, color=palette[k], alpha=0.2, linewidth=0)

# Highlight analysis window used for PCA/KMeans
plt.axvspan(win_start_ms, win_end_ms, color="lightgreen", alpha=0.15, zorder=0)
plt.axvline(0, color="k", lw=1, alpha=0.4)
plt.xlabel("Time (ms)")
plt.ylabel("Probability density")
plt.title(
    f"Class-averaged responses (PDF normalized in 0-300ms; "
    f"analysis window {win_start_ms}-{win_end_ms}ms)"
)
plt.legend(frameon=False, loc="best")
plt.tight_layout()

out_png2 = os.path.join(out_dir, f"{brain_area}_cluster_avg_pdf_0to300ms.png")
plt.savefig(out_png2, dpi=200)
print(f"Saved: {out_png2}")
plt.show()

#%%


