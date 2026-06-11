''''
Getting Face selective cells in AL.

'''

#%%
import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np
import copy


ceiled_files = r'D:\_DataTemp\data_tmp\ASB_AL_Ceiled.pkl'



#%%
c_psth,c_info = ot.Load_Variable(ceiled_files)
c_body = c_info[c_info['Contra']=='Face_Contra'].reset_index(drop=True)
plt.hist(c_body['D_Prime'])


face_ids = list(c_body[c_body['D_Prime']>0.5]['Cell'])

al_cells = c_psth[face_ids,:,:]
ot.Save_Variable(r'D:\_DataTemp\data_tmp','AL_Cells',al_cells)
#%%
redplot = Redplot(al_cells)
N_cell = len(redplot)

#%%
avr_resp = redplot[:,300:1300].reshape(N_cell,5,200).mean(1)


new_ids = []
for i in range(5):# animate parts 
    c_list = np.arange(40*i,40*i+20)
    new_ids.extend(c_list)

for i in range(5):# inanimate parts
    c_list = np.arange(40*i+20,40*i+40)
    new_ids.extend(c_list)
new_ids = np.array(new_ids)

arranged_redplot = avr_resp[:,new_ids]

# sns.heatmap(arranged_redplot,center=0,cmap='bwr',vmax=0.5,vmin=-0.5)
for i in range(N_cell):
    c_sum = arranged_redplot[i,:].mean()
    c_std = arranged_redplot[i,:].std()
    arranged_redplot[i,:]=(arranged_redplot[i,:]-c_sum)/c_std
# then sort y axis.
PC_Comps,point_coords,pca = Do_PCA(arranged_redplot,'Cell',1)
y_ids = np.argsort(PC_Comps[0,:])
arranged_redplot = arranged_redplot[y_ids,:]
#%% Plot
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(3,5),dpi=240)
sns.heatmap(arranged_redplot,cmap='bwr',vmax=3,vmin=-3,center=0,cbar=False,ax=ax)
# ax.set_xticks(np.arange(0,210,10))
# ax.set_xticklabels(['','Raw','','C4','','C3','','C2','','C1','','Raw','','C4','','C3','','C2','','C1',''],size=6)
ax.set_xticks(np.arange(0,200,20)+10)
ax.set_xticklabels(['Raw','C4','C3','C2','C1','Raw','C4','C3','C2','C1'],size=6)
ax.set_yticks([0,50,100,150])


