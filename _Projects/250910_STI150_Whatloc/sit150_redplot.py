

#%%


import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np


wp=r'D:\#Data\STI150_2'
savepath = wp

gn_name = ot.Get_File_Name(wp,'.mat')[0]


#%% calculate psth and 
c_name = gn_name.split('\\')[-1][9:-4]
c_psth = PSTH_From_Goodunit(gn_name,img_num=72)
np.save(ot.Join(savepath,c_name+"_PSTH"),c_psth)
# set response.
redplot = Redplot(c_psth.mean(1))
#%%
fig,ax = plt.subplots(ncols=1,nrows=1,dpi=240,figsize=(3,5))
sns.heatmap(redplot[:,:40],center=0,vmax=1,vmin=-1,ax=ax,cbar=False)
# ax.invert_yaxis()
#%%
plt.plot(redplot[350:,:].mean(0))