'''

We will try to explain Raw response use S4 data. 

1 - Time-windowed response of S4 data.
2 - PCA of S4 data, getting (10 or 90% var) comps, these comps are pattern of response 
3 - Use this patterns to fit raw response.
4 - Validate R2 of different time windows
5 - Extract whether there are unique patterns in Raw, which cannot be explained by S4 data.
'''


#%%
from Spike_Tools import *
from Py_Structure.Info_Files.InfoLoader import Load_Info
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageEnhance
from Py_Structure.Info_Files.Stim_ID_List import Stim_ID
from sklearn.decomposition import PCA

wp = r'E:\#Preprocessed_Data\Selected_Cells'

areas = ['MSB','ASB','ML','AL']
used_aras = areas[0]
used_rsps = np.load(ot.Join(wp,f'{used_aras}_Cells_Metamer_Only.npz'))['psth']
Stim_Caller = Stim_ID(stim_type='Metamer_Raw')
metamer_ids = Stim_Caller.Stim_Conditions

resp_t = used_rsps[:, :, 100:400]
den = resp_t.sum(axis=(1, 2), keepdims=True)
psth_pdf = np.divide(resp_t, den, out=np.zeros_like(resp_t, dtype=float), where=den > 0) * 1000

#%% ###### time-win bin for pdf, then do pca.
bin_ms = 10

time_len = psth_pdf.shape[-1]
n_bins = time_len // bin_ms
psth_pdf_10ms = psth_pdf.reshape(psth_pdf.shape[0], psth_pdf.shape[1], n_bins, bin_ms).sum(axis=-1)
# psth_pdf_flat = psth_pdf_10ms.reshape(psth_pdf_10ms.shape[0], -1)

#%%
s4_ids = np.where(metamer_ids.Shuffle_Level == 4)[0]
raw_ids = np.where(metamer_ids.Shuffle_Level == 0)[0]
### PCA to get response patterns in S4 response 
pdf_s4 = psth_pdf_10ms[:,s4_ids,:].reshape(psth_pdf_10ms.shape[0], -1)
pdf_raw = psth_pdf_10ms[:,raw_ids,:].reshape(psth_pdf_10ms.shape[0], -1)

#%% ### Build a PCA model based on s4 data.
# Treat each "image x time-bin" as one sample, and neurons as features.
n_cells = psth_pdf_10ms.shape[0]
n_s4_imgs = len(s4_ids)
n_raw_imgs = len(raw_ids)

X_s4 = psth_pdf_10ms[:, s4_ids, :].transpose(1, 2, 0).reshape(-1, n_cells)   # (n_s4_imgs*30, n_cells)
X_raw = psth_pdf_10ms[:, raw_ids, :].transpose(1, 2, 0).reshape(-1, n_cells)  # (n_raw_imgs*30, n_cells)

n_pc = 100
pca_s4 = PCA(n_components=n_pc)
pc_loadings_6000 = pca_s4.fit_transform(X_s4)   # (n_s4_imgs*30, 10), each row is one sample loading
pc_neuron_weights = pca_s4.components_          # (10, n_cells), each row is one PC's neuron weights

# Optional: project raw response into the same PC space learned from S4.
pc_loadings_raw = pca_s4.transform(X_raw)       # (n_raw_imgs*30, 10)

# Reshape loadings back to (image, time-bin, pc), then average over images for temporal curves.
pc_loadings_s4_3d = pc_loadings_6000.reshape(n_s4_imgs, n_bins, n_pc)
pc_time_curve = pc_loadings_s4_3d.mean(axis=0)  # (30, 10): time curve of each PC

# Friendly tables for quick inspection.
pc_weight_df = pd.DataFrame(
    pc_neuron_weights.T,
    index=[f"cell_{i}" for i in range(n_cells)],
    columns=[f"PC{i+1}" for i in range(n_pc)],
)
pc_time_curve_df = pd.DataFrame(
    pc_time_curve,
    index=[f"bin_{i}" for i in range(n_bins)],
    columns=[f"PC{i+1}" for i in range(n_pc)],
)

print(f"X_s4 shape: {X_s4.shape}")
print(f"pc_neuron_weights shape (PC x cell): {pc_neuron_weights.shape}")
print(f"pc_loadings_6000 shape (sample x PC): {pc_loadings_6000.shape}")
print(f"pc_time_curve shape (time_bin x PC): {pc_time_curve.shape}")
print(f"Explained variance ratio (10 PCs): {pca_s4.explained_variance_ratio_.sum():.4f}")
