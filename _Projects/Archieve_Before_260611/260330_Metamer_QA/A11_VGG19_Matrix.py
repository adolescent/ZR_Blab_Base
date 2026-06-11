"""
Calculate VGG19 RSA and averaged response correlation matrix.
"""

#%%
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Matrix_Tools import Corr_Matrix
from tqdm import tqdm

import OS_Tools as ot

vgg19_path = r"E:\#Preprocessed_Data\260305_Report_Data\VGG19_rsps"
savepath = vgg19_path
n_img = 1000

vgg_rsps = np.load(ot.Join(vgg19_path, "VGG19_Response.npz"), allow_pickle=True)


#%%
# Use only the two requested layers.
layer_map = {
    "last_conv": vgg_rsps["last_conv"].reshape(n_img, -1),
    "fc1": vgg_rsps["fc1"].reshape(n_img, -1),
}
print(f"last_conv shape: {layer_map['last_conv'].shape}")
print(f"fc1 shape: {layer_map['fc1'].shape}")


#%%
# Pairwise correlation inside each source image (25 variants x 40 images).
img_counter = 0
for c_layer, c_response in tqdm(layer_map.items(), desc="Layer corr"):
    data_reshaped = c_response.reshape(25, 40, c_response.shape[1])

    for img_idx in range(40):
        img_data = data_reshaped[:, img_idx, :]
        corr_matrix = np.corrcoef(img_data)
        results = []
        rows, cols = np.triu_indices(25, k=1)

        for r, c in zip(rows, cols):
            results.append(
                {
                    "Network": "VGG19",
                    "Layer": c_layer,
                    "Img_Index": img_idx,
                    "C_img1": r,
                    "C_img2": c,
                    "Corr": corr_matrix[r, c],
                    "Dist": 1 - corr_matrix[r, c],
                }
            )
        df = pd.DataFrame(results)

        if img_counter == 0:
            Network_Corr = copy.deepcopy(df)
        else:
            Network_Corr = pd.concat([Network_Corr, df], ignore_index=True)
        img_counter += 1

Network_Corr.to_parquet(ot.Join(savepath, "VGG19_Corr.parquet"), index=False)
print(f"Saved: {ot.Join(savepath, 'VGG19_Corr.parquet')}")
print("Rows:", len(Network_Corr), "Layers:", Network_Corr["Layer"].unique())


#%% ################## Plot 1, RSA MATRIX (last_conv) ####################
plotable = Corr_Matrix(layer_map["fc1"].T, fill_diag=False)[:200, :200]

fig, ax = plt.subplots(ncols=1, nrows=1, dpi=240, figsize=(5, 5))
sns.heatmap(
    plotable,
    ax=ax,
    cbar=False,
    square=True,
    xticklabels=False,
    yticklabels=False,
    cmap="bwr",
    center=0,
)


#%% ################## Plot 2, Averaged Constrain Corr (fc1) ####################
plotable = Network_Corr[Network_Corr.Layer == "fc1"]

df_mirror = plotable.copy()
df_mirror.columns = ["Network", "Layer", "Img_Index", "C_img2", "C_img1", "Corr", "Dist"]
df_total = pd.concat([plotable, df_mirror], axis=0, ignore_index=True)

df_total["C_img1"] = df_total["C_img1"] % 5
df_total["C_img2"] = df_total["C_img2"] % 5

matrix = df_total.pivot_table(
    index="C_img1",
    columns="C_img2",
    values="Corr",
    aggfunc="mean",
)

fig, ax = plt.subplots(ncols=1, nrows=1, dpi=240, figsize=(5, 5))
sns.heatmap(
    matrix,
    annot=False,
    cmap="RdBu_r",
    center=0,
    vmax=1,
    cbar=False,
    xticklabels=False,
    yticklabels=False,
    square=True,
    ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("")
plt.show()

