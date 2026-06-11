'''
This script will compare stimulus onset response of all 5 conditions, comparing it with FOB.

'''

#%%

import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np
import copy

# c_psth,c_info = ot.Load_Variable(r'D:\_DataTemp\Metamer\ceiled_response\MSB\250829_ZhuangZhuang_Metamer_Pool4_C4321_Object_1k+FOB_g2_MSB_PSTH_Ceiled.pkl')
c_psth,c_info = ot.Load_Variable(r'D:\_DataTemp\Metamer\ceiled_response\ASB\250828_JianJian_Metamer_Pool4_C4321_Object_1k+FOB_g6_AL_ASB_PSTH_Ceiled.pkl')


# sort cell 
c_info = c_info[c_info['Contra']=='Body_Contra'].reset_index(drop=True)
body_dp = np.array(c_info['D_Prime'])

body_dp_sorted = np.sort(body_dp)
sorted_indices = np.argsort(body_dp)
sorted_psth = c_psth[sorted_indices]

used_cells = sorted_psth[np.where(body_dp_sorted>0.5)[0][0]:,:]

#%%
body_resp = used_cells[:,:24,:].reshape(len(used_cells),24,90,5).mean(-1)
metamer_resp = used_cells[:,72:,:].reshape(len(used_cells),1000,90,5).mean(-1)

id_lists = ['Raw_ani','Raw_inani','C4_ani','C4_inani','C3_ani','C3_inani','C2_ani','C2_inani','C1_ani','C1_inani']
all_resp = pd.DataFrame(columns=['Cell','Type','Time','Response'],index=np.arange(100000000))

counter=0
for i in tqdm(range(len(used_cells))):
    for j in range(1000):
        ids = (j+1)%200
        if ids == 0:
            c_type = id_lists[-1]
        else:
            c_type = id_lists[ids//20]
        for k in range(90):
            all_resp.loc[counter,:] = [i,c_type,k,metamer_resp[i,j,k]]
            counter+=1

for i in tqdm(range(len(used_cells))):
    for j in range(24):
        for k in range(90):
            all_resp.loc[counter,:] = [i,'Body',k,body_resp[i,j,k]]
            counter+=1
#%%
plotable = all_resp.dropna()
# plotable = plotable[plotable['Type'].isin(['Body','Raw_ani','C4_ani','C3_ani','C2_ani','C1_ani'])]
plotable = plotable[plotable['Type'].isin(['Raw_inani','C4_inani','C3_inani','C2_inani','C1_inani'])]
#%%
fig,ax = plt.subplots(ncols=1,nrows=1,dpi=240,figsize=(6,4))
# sns.lineplot(data = plotable,x='Time',y='Response',hue='Type',alpha=0.7,ax=ax,palette='tab10',hue_order=['Body','Raw_ani','C4_ani','C3_ani','C2_ani','C1_ani'])
sns.lineplot(data = plotable,x='Time',y='Response',hue='Type',alpha=0.7,ax=ax,palette='tab10',hue_order=['Raw_inani','C4_inani','C3_inani','C2_inani','C1_inani'])
ax.set_xticks([0,20,40,60,80])
ax.set_xticklabels([-100,0,100,200,300])
