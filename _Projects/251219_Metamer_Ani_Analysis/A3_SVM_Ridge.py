'''

以下两部分分析：


1.双路ridge回归，找到每个神经元对整体解码和局部特征解码的贡献程度，评估每个神经元在局部和整体特征解码中的正确率

2.


'''

#%%

# from Cell_Class import Cell_Infos
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
from tqdm import tqdm

wp = r'E:\#Preprocessed_Data\Selected_Cells'
result_path = r'E:\#Coding_traces\251219_Metamer_Ani_Only_Anis'


#%% 
'''
载入数据
'''

msb_infos = np.load(ot.Join(wp,'MSB_Cells_Metamer_Only.npz'),allow_pickle=True)
asb_infos = np.load(ot.Join(wp,'ASB_Cells_Metamer_Only.npz'),allow_pickle=True)

## cut data, getting animated only,and save animate matrix.
msb_resps = msb_infos['psth']
n_msb = msb_resps.shape[0]
temp_data = msb_resps.reshape(n_msb,25,40, 450)
msb_ani_only = temp_data[:, :, :20, :].reshape(n_msb, -1, 450)
msb_ani_avr = msb_ani_only[:,:,160:320].sum(-1)

np.savez_compressed(ot.Join(wp,'MSB_Metamer_Ani_only.npz'),psth=msb_ani_only)


asb_resps = asb_infos['psth']
n_asb = asb_resps.shape[0]
temp_data = asb_resps.reshape(n_asb, 25, 40, 450)
asb_ani_only = temp_data[:, :, :20, :].reshape(n_asb, -1, 450)
asb_ani_avr = asb_ani_only[:,:,160:320].sum(-1)
np.savez_compressed(ot.Join(wp,'ASB_Metamer_Ani_only.npz'),psth=asb_ani_only)

## if you need cell ids
msb_dps = pd.DataFrame(msb_infos['d_primes'],columns=['Cell','D_Prime','Category','Loc','Cell_ID'])
msb_fob = pd.DataFrame(msb_infos['response'],columns=['Cell','Category','Response','Loc','Cell_ID'])
asb_dps = pd.DataFrame(asb_infos['d_primes'],columns=['Cell','D_Prime','Category','Loc','Cell_ID'])
asb_fob = pd.DataFrame(asb_infos['response'],columns=['Cell','Category','Response','Loc','Cell_ID'])

#%%
'''
双路Ridge回归，判断每个cell在两个任务中的影响因素
'''


