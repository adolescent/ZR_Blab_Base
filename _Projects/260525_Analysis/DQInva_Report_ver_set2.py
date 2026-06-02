'''
Re-Analyze the DQInva data with Stimset2.
'''

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
save_path = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable'
set2_stims = np.load(ot.Join(save_path,'set2_rsp_z_sorted.npy'),allow_pickle=True)

#%% RSA: 240×240 (reordered: Sh–Tx–ShC–TxC per obj block)
# orig sorted 80: Sh | ShC | Tx | TxC  →  plot: Sh | Tx | ShC | TxC
_blk = lambda s: np.r_[s + np.arange(20), s + 40 + np.arange(20),
                       s + 20 + np.arange(20), s + 60 + np.arange(20)]
idx = np.concatenate([_blk(s) for s in (0, 80, 160)])
rsa = np.corrcoef(set2_stims[:, idx].T)
np.fill_diagonal(rsa, 0)

_sub_lbl = ['B', 'F', 'Fr']          # Body, Face, Fruit
_cond_lbl = ['Sh', 'Tx', 'SC', 'TC']  # Shading, Texture, Shading_CTR, Tex_CTR

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(rsa, vmin=-0.5, vmax=0.5, center=0, cmap='RdBu_r',
            xticklabels=False, yticklabels=False, square=True,
            cbar_kws={'shrink': 0.7, 'label': 'Pearson r'}, ax=ax,cbar=False)
for x in (80, 160):
    ax.axvline(x, color='yellow', lw=2)
    ax.axhline(x, color='yellow', lw=2)
for s in (0, 80, 160):
    for x in (s + 20, s + 40, s + 60):
        ax.axvline(x, color='k', lw=0.8, ls='--')
        ax.axhline(x, color='k', lw=0.8, ls='--')
for i, sl in enumerate(_sub_lbl):
    ax.text(40 + i * 80, 1.06, sl, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    for j, cl in enumerate(_cond_lbl):
        ax.text(10 + i * 80 + j * 20, 1.01, cl, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7)
fig.tight_layout()

#%%


