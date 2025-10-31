'''
This script will rerun several graphs in graphs in NI article.

'''

#%%
import Common_Functions.OS_Tools as ot
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

spon_before = ot.Load_Variable(r'D:\_Codes\ZR_Blab_Base\_Projects\251020_Report_Related\Spon_Before.pkl')

#%%
plt.clf()

vmax = 4
vmin = -2
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,3),dpi = 300,sharex = True)
label_size = 10


# Plot Core Heatmap
axes[0].plot(np.array(spon_before.iloc[4700:4960,200]))
sns.heatmap(spon_before.iloc[4700:4960,:].T,center = 0,xticklabels=False,yticklabels=False,ax = axes[1],vmax = vmax,vmin = vmin,cbar= False,cmap = 'bwr')


# set time scale in seconds
fps = 1.301
# axes[1].set_xticks([0*fps,100*fps,200*fps,300*fps,400*fps,500*fps])
# axes[1].set_xticklabels([0,100,200,300,400,500],fontsize = label_size)
axes[1].set_xticks([0*fps,100*fps,200*fps])
axes[1].set_xticklabels([0,100,200],fontsize = label_size)
axes[0].set_yticks([])
fig.tight_layout()
plt.show()

