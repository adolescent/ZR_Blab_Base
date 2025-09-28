'''
Re-arrange response of all cell's MSB and ASB response.

'''



#%%
import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np
import copy

wp=r'D:\#Data\Metamer\ceiled_response\ASB'

body_dp_sorted,redplot,sorted_psth = ot.Load_Variable(wp,'Sorted_Response.pkl')

N_cell = len(redplot)
#%% re-arrange redplot.
avr_resp = redplot.reshape(N_cell,5,200).mean(1)

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
ax.set_yticks([0,100,200,300])
