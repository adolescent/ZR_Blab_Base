'''
Keep only odd&end>0.3 cells as valid.

Then generate psth and FOB preference index of each cell.

'''


#%%
import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np


psth_path = r'D:\#Data\Metamer\PSTH'
savepath = r'D:\#Data\Metamer\ceiled_response'

gn_paths = ot.Get_File_Name(psth_path,'.npy')
# first 2 use fob 72, others use sti150 as localizer.


#%% cycle this part MANUALLY
c_gn = gn_paths[2]
exp_psth = np.load(c_gn)
c_name = c_gn.split('\\')[-1][:-4]

ceiling_index = odd_end_ceiling(exp_psth[:,:,1000:,:],np.arange(150,250))
ok_cells = np.where(ceiling_index>0.3)[0]

ceiled_response = exp_psth.mean(1)[ok_cells,:,:] #average between blks
redplot = Redplot(ceiled_response,np.arange(75,125),np.arange(150,250))

sns.heatmap(redplot[:,:1000],center=0)
#%% calculate fob tuning 
localizer_info = pd.read_csv(r'D:\#Data\Localizer_infos\FOB72.csv',index_col=0)
tuning_frame = Calculate_Cell_Tunings(ceiled_response[:,1000:,:],localizer_info)

#%%% save part
ot.Save_Variable(savepath,c_name+'_Ceiled',(ceiled_response,tuning_frame))

