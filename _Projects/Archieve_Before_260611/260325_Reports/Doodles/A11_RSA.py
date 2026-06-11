


#%%
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
import copy
import warnings
import gc
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB','.joblib')
asb_rsp = np.load(r'E:\#Preprocessed_Data\Selected_Cells\ASB_Cells_Doodle.npz')['psth'][:,:,160:320].sum(-1)
msb_rsp = np.load(r'E:\#Preprocessed_Data\Selected_Cells\MSB_Cells_Doodle.npz')['psth'][:,:,160:320].sum(-1)
ml_rsp = np.load(r'E:\#Preprocessed_Data\Selected_Cells\ML_Cells_Doodle.npz')['psth'][:,:,160:320].sum(-1)

avr_asb = asb_rsp/asb_rsp.std(1,keepdims=1)
avr_msb = msb_rsp/msb_rsp.std(1,keepdims=1)
avr_ml = ml_rsp/ml_rsp.std(1,keepdims=1)
#%%
'''Plot single area response'''
from scipy.stats import zscore

plotable = avr_ml
# Perform Z-score normalization across each row (cell), then clip
plotable_z = zscore(plotable, axis=1, ddof=1)
doodle = plotable_z[:, :100]
real = plotable_z[:, 800:900]

fig,ax = plt.subplots(ncols=1,nrows=1,dpi=300,figsize=(2,2))
sns.heatmap(
    doodle,
    ax=ax,
    cmap='bwr',
    center=0,
    cbar=False,
    cbar_kws={"fraction":0.046, "pad":0.04},xticklabels=False,yticklabels=False,vmax =5,vmin=-5
   
)
# ax.set_title('ML')
# ax.set_xlabel('Stimulus')
# ax.set_ylabel('Cell')
plt.show()


#%%


plotable = avr_ml

# Perform Z-score normalization across each row (cell), then clip
plotable_z = zscore(plotable, axis=1, ddof=1)
# plotable_z = np.clip(plotable_z, 0, 7)

doodle = plotable_z[:, :400]
real = plotable_z[:, 800:1200]

import seaborn as sns

# Plot doodle and real using seaborn heatmap, with center=0
fig, axes = plt.subplots(2, 1, figsize=(5,5), sharey=True)

sns.heatmap(
    doodle,
    ax=axes[0],
    cmap='bwr',
    center=0,
    cbar=False,
    cbar_kws={"fraction":0.046, "pad":0.04},xticklabels=False,yticklabels=False,vmax =5,vmin=-5
)
axes[0].set_title('Doodle')
axes[0].set_xlabel('Stimulus')
axes[0].set_ylabel('Cell')

sns.heatmap(
    real,
    ax=axes[1],
    cmap='bwr',
    center=0,
    cbar=False,
    cbar_kws={"fraction":0.046, "pad":0.04},xticklabels=False,yticklabels=False,vmax =5,vmin=-5
   
)
axes[1].set_title('Real')
axes[1].set_xlabel('Stimulus')
axes[1].set_ylabel('Cell')

plt.tight_layout()
plt.show()

# Similarity between the two matrices
doodle_flat = doodle.ravel()
real_flat = real.ravel()

pearson_r = np.corrcoef(doodle_flat, real_flat)[0, 1]
cosine_sim = np.dot(doodle_flat, real_flat) / (
    np.linalg.norm(doodle_flat) * np.linalg.norm(real_flat) + 1e-12
)
fro_dist = np.linalg.norm(doodle - real, ord='fro')

print(f'Pearson r: {pearson_r:.4f}')
print(f'Cosine similarity: {cosine_sim:.4f}')
print(f'Frobenius distance: {fro_dist:.4f}')


# %%
