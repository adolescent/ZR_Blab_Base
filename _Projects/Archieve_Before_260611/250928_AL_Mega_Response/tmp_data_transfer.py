''''
This is a temporary analysis for data transfer, getting AL cells and their response.

'''


#%%
'''
Step1, transfer good unit into psth file.
'''

import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np


# wp=r'D:\#Data\Metamer'
c_gn = r'D:\_DataTemp\data_tmp\GoodUnit_250922_JianJian_Mega_Metamer_v250920_g0_AL_ASB.mat'
# c_gn = r'D:\_DataTemp\data_tmp\GoodUnit_250925_ZhuangZhuang_Mega_Metamer_v250920_g2_MSB.mat'

savepath=r'D:\_DataTemp\data_tmp'

c_name = c_gn.split('\\')[-1][9:-4]
c_psth = PSTH_From_Goodunit(c_gn,img_num=2240)
np.save(ot.Join(savepath,c_name+"_PSTH"),c_psth)

#%%
'''
Nois ceiling, we find that noise ceiling show different property between ASB and MSB..
'''

c_msb = np.load(r'D:\_DataTemp\data_tmp\250925_ZhuangZhuang_Mega_Metamer_v250920_g2_MSB_PSTH.npy')
c_asbal = np.load(r'D:\_DataTemp\data_tmp\250922_JianJian_Mega_Metamer_v250920_g0_AL_ASB_PSTH.npy')



ceiling_index_msb = odd_end_ceiling(c_msb[:,:6,:300,:],np.arange(150,250))
ceiling_index_asb = odd_end_ceiling(c_asbal[:,:,:300,:],np.arange(150,250))

sns.histplot(ceiling_index_msb,alpha=0.7)
sns.histplot(ceiling_index_asb,alpha=0.7)
#%% calculate average response. In Hz.

msb_response_avr = c_msb.mean(1)[:,:300,100:300].sum(-1)*5
asb_response_avr = c_asbal.mean(1)[:,:300,100:300].sum(-1)*5

sns.histplot(msb_response_avr.mean(0),alpha=0.7,bins=np.arange(2,8.5,0.5))
sns.histplot(asb_response_avr.mean(0),alpha=0.7,bins=np.arange(2,8.5,0.5))


#%% for this case, ASB-AL use 0.15 for noise ceiling, MSB use 0.3.

ceiling_index_asb = odd_end_ceiling(c_asbal[:,:,:300,:],np.arange(150,250))
ok_cells_asb = np.where(ceiling_index_asb>0.15)[0]
print(len(ok_cells_asb))
ceiled_response_asb = c_asbal.mean(1)[ok_cells_asb,:,:] #average between blks
redplot = Redplot(ceiled_response_asb,np.arange(75,125),np.arange(150,250))

sns.heatmap(redplot[:,:1300],center=0)

localizer_info = pd.read_csv(r'D:\_Codes\ZR_Blab_Base\_Projects\250912_Cell_Screening\STI150.csv',index_col=0)
tuning_frame = Calculate_Cell_Tunings(ceiled_response_asb[:,:300,:],localizer_info)

ot.Save_Variable(savepath,'ASB_AL_Ceiled',(ceiled_response_asb,tuning_frame))

#%% save msb

ceiling_index_msb = odd_end_ceiling(c_msb[:,:,:300,:],np.arange(150,250))
ok_cells_msb = np.where(ceiling_index_msb>0.3)[0]
print(len(ok_cells_msb))
ceiled_response_msb = c_msb.mean(1)[ok_cells_msb,:,:] #average between blks
redplot = Redplot(ceiled_response_msb,np.arange(75,125),np.arange(150,250))

sns.heatmap(redplot[:,:1300],center=0)

localizer_info = pd.read_csv(r'D:\_Codes\ZR_Blab_Base\_Projects\250912_Cell_Screening\STI150.csv',index_col=0)
tuning_frame = Calculate_Cell_Tunings(ceiled_response_msb[:,:300,:],localizer_info)

ot.Save_Variable(savepath,'MSB_Ceiled',(ceiled_response_msb,tuning_frame))



