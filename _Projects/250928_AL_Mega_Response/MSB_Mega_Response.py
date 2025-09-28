

#%%
import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np
import copy


ceiled_files = r'D:\_DataTemp\data_tmp\MSB_Ceiled.pkl'

#%%
c_psth,c_info = ot.Load_Variable(ceiled_files)
c_body = c_info[c_info['Contra']=='Body_Contra'].reset_index(drop=True)
plt.hist(c_body['D_Prime'])


face_ids = list(c_body[c_body['D_Prime']>0.5]['Cell'])

al_cells = c_psth[face_ids,:,:]
ot.Save_Variable(r'D:\_DataTemp\data_tmp','MSB_Cells',al_cells)

#%%
redplot = Redplot(al_cells[:,300:,:])
N_cell = len(redplot)
#%%

used_plots = redplot[:,800:-400]
arranged_redplot = np.zeros(shape = used_plots.shape)
N_cell=len(used_plots)
for i in range(N_cell):
    c_sum = used_plots[i,:].mean()
    c_std = used_plots[i,:].std()
    arranged_redplot[i,:]=(used_plots[i,:]-c_sum)/c_std
PC_Comps,point_coords,pca = Do_PCA(arranged_redplot,'Cell',1)
y_ids = np.argsort(PC_Comps[0,:])
arranged_redplot = arranged_redplot[y_ids,:]
#%%

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6,5),dpi=240)
sns.heatmap(arranged_redplot,cmap='bwr',vmax=3,vmin=-3,center=0,cbar=False,ax=ax)

ax.set_xticks([0,200,300,340,500,660,740])
ax.set_yticks([0,100,200,300])
