

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

site = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\DQInva_video','.joblib')[0]
a = JL.load(site)

stim_info = a.stim_info
#%%
ani_cells,ani_psth = a.Cell_Selection(ceiling=0.2,prefer='Animate',dp_thres=0.5)
len(ani_cells)
redplot = ani_psth[:,:,1050:2500].sum(-1) # -1000~3500
redplot_z = (redplot-redplot.mean(1,keepdims=True))/redplot.std(1,keepdims=True)
#%%
avr_rsp = redplot[:,72:].reshape(len(redplot),3,-1).mean(1)
avr_rsp_z = (avr_rsp-avr_rsp.mean(1,keepdims=True))/avr_rsp.std(1,keepdims=True)

cat_n = stim_info.iloc[72:150].Category.value_counts().reindex(['Body', 'Face', 'Chair', 'Tool'])
cat_divs = cat_n.cumsum().iloc[:-1].tolist()  # 19, 39, 58

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(avr_rsp_z, center=0, vmax=3, vmin=-3, cmap='RdBu_r',
            xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'z-scored response', 'shrink': 0.8}, ax=ax)
for x in cat_divs:
    ax.axvline(x, color='yellow', lw=2)
ax.set_xlabel('Body | Face | Chair | Tool')
ax.set_ylabel('Neuron')
fig.tight_layout()
np.save(ot.Join(save_path,'dq_video_BFCT.npy'),avr_rsp_z)

#%% Rank 78 BFCT objects by mean response
import re

info_bfct = stim_info.iloc[72:150].reset_index(drop=True)  # Cycle1, 19+20+19+20
obj_ids = (info_bfct.Category.str.lower() + '_' +
           info_bfct.FileName.str.extract(r'(\d+)', expand=False))

m = avr_rsp_z.mean(0)
rsp_ord = np.argsort(m)[::-1]  # strong → weak
sorted_obj_ids = obj_ids.iloc[rsp_ord].tolist()

rsp_rank = pd.DataFrame({
    'rank': np.arange(1, len(rsp_ord) + 1),
    'obj_id': sorted_obj_ids,
    'mean_rsp': m[rsp_ord],
    'col_idx': rsp_ord,
})

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(m[rsp_ord], lw=2)
ax.set_xlabel('Stimulus rank (strong → weak)')
ax.set_ylabel('Mean z-scored response')
fig.tight_layout()



