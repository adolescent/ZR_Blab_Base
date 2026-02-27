'''
提取全部的doodle刺激集的MSB神经元，得到一个stimeset。
'''

#%%
# from Cell_Class import Cell_Infos
import mat73
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
import warnings
import copy

warnings.filterwarnings("ignore")
wp=r'E:\#Preprocessed_Data\SiteClass\Doodle'
save_path = r'E:\#Preprocessed_Data\Selected_Cells'

all_site_path = ot.Get_File_Name(wp,'.joblib')
#%% 拼合所有位置的满足条件的神经元。
cell_counter = 0
for i,cloc in tqdm(enumerate(all_site_path)):

    a = JL.load(cloc)
    c_info = a.Site_Info
    c_loc = a.site_name
    rec_site = a.brain_areas
    if 'ASB' not in rec_site:
        print('No ASB. Continue')
        continue

    ok_cells = c_info[c_info.Ceiling_Index>0.3]
    ok_cells = ok_cells[ok_cells.Best_D_Prime>0.5]
    tuned_cells = np.array(ok_cells[ok_cells.Best_Prefer=='Body'].Cell)

    tuned_resps = a.avr_psth[tuned_cells,:,:]


    if a.stimset == 'Doodle_v260119' or a.stimset == 'Doodle_v260121':
        selected_stim = tuned_resps
    else:
        print(f'Location {cloc} is not a doodle site.')
        continue

    # generate concated response.
    try:# If no var, init.
        all_tuned_resp = np.concatenate([all_tuned_resp,selected_stim],axis=0)
    except:
        all_tuned_resp = copy.deepcopy(selected_stim)

    # generate cell infos.
    c_dps = a.Cell_FOB_DPrimes
    c_rsp = a.Cell_FOB_Response_avr
    c_dps = c_dps[c_dps['Cell'].isin(tuned_cells)] # only body-selected part
    c_rsp = c_rsp[c_rsp['Cell'].isin(tuned_cells)]

    # add loc and stim.
    c_dps['Loc'] = c_loc
    c_rsp['Loc'] = c_loc
    c_dps['Stimset'] = a.stimset
    c_rsp['Stimset'] = a.stimset

    try:
        all_d_primes = pd.concat((all_d_primes,c_dps))
        all_response = pd.concat((all_response,c_rsp))
    except:
        all_d_primes = copy.deepcopy(c_dps)
        all_response = copy.deepcopy(c_rsp)
        
    del a

all_d_primes['Cell_ID'] = all_d_primes.groupby(['Loc', 'Cell'], sort=False).ngroup()
all_response['Cell_ID'] = all_response.groupby(['Loc', 'Cell'], sort=False).ngroup()


#%% save all msb cells
np.savez_compressed(ot.Join(save_path,'Doodle_All_ASB_Response.npz'),psth = all_tuned_resp,d_primes = all_d_primes,response = all_response)

