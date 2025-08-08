'''
Calculate quicker d prime subtraction map online.
Maybe a little quicker?


'''
#%%

from toolkits import *

wp=r'D:\#Data\Loc_Example\test_wp'
npx_path = r'D:\#Data\Loc_Example\npx_root_decoy\NPX_MD241029_exp_g0'


# thres=1.5
# base_time = np.arange(75,125) # -25~25ms
# onset_time = np.arange(150,250) # 50~150ms

ov = Online_Viewer(wp,npx_path)
ov.Process()
# ov.Plot_Global()

#%% d prime graphs 
print(ov.clusts)
A_set = [6]
B_set = [1,2,3,4,5]
cc_resp = ov.Subtract(A_set,B_set,plot=True)
#%% plot sorted cell by subtraction response.
ov.Plot_Sorted_Global(cc_resp)
