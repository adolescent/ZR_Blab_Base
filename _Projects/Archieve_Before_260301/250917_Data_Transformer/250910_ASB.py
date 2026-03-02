'''
Transfer data into processable npy file, and getting it's ceiled response.

'''






#%%
import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np


# wp=r'D:\#Data\Metamer'
c_gn = r'D:\#Data\Metamer\GoodUnit_250910_JianJian_Metamer_P4_C4321_Object_STI150_1300_g1_ASB.mat'
savepath=r'D:\#Data\Metamer\PSTH'

c_name = c_gn.split('\\')[-1][9:-4]
c_psth = PSTH_From_Goodunit(c_gn,img_num=1300)
np.save(ot.Join(savepath,c_name+"_PSTH"),c_psth)


#%% odd&end ceil cell response.

ceiling_index = odd_end_ceiling(c_psth[:,:,1000:,:],np.arange(150,250))
ok_cells = np.where(ceiling_index>0.3)[0]
ceiled_response = c_psth.mean(1)[ok_cells,:,:] #average between blks
redplot = Redplot(ceiled_response,np.arange(75,125),np.arange(150,250))

sns.heatmap(redplot[:,:1000],center=0)

#%% calculate fob tuning 
localizer_info = pd.read_csv(r'D:\#Data\Localizer_infos\STI150.csv',index_col=0)
tuning_frame = Calculate_Cell_Tunings(ceiled_response[:,1000:,:],localizer_info)

ot.Save_Variable(savepath,c_name+'_Ceiled',(ceiled_response,tuning_frame))
#%% sorting by body d prime.
c_body = tuning_frame[tuning_frame['Contra']=='Body_Contra'].reset_index(drop=True)


body_dp = np.array(c_body['D_Prime'])
body_dp_sorted = np.sort(body_dp)
sorted_indices = np.argsort(body_dp)
sorted_psth = ceiled_response[sorted_indices]

redplot = Redplot(sorted_psth)


