'''
This script will select good body cells in MSB.

NOTE only body cells, and only metamer selected. 

'''


#%%

import mat73
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
import copy

msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB','.joblib')
save_path = r'E:\#Preprocessed_Data\Selected_Cells'

#%%
# cell_counter = 0
for i,cloc in tqdm(enumerate(msb_sites)):
    a = JL.load(cloc)
    c_info = a.Site_Info
    c_loc = a.site_name

    ok_cells = np.array(c_info[c_info.Ceiling_Index>0.3].Cell)
    all_cell_dps = a.Cell_FOB_DPrimes.pivot(index='Cell',columns='Category',values='D_Prime')
    body_cells = np.array(all_cell_dps[all_cell_dps['Face']>0.5].index)
    body_cells = np.intersect1d(ok_cells, body_cells)

    # body_cells = np.array(ok_cells[ok_cells.Best_D_Prime>0.5].Cell)
    body_resps = a.avr_psth[body_cells,:,:]
    raw_rsp = a.avr_psth
    if a.stimset == 'Metamer1072':
        metamers = body_resps[:,72:,:]
    elif a.stimset == 'Metamer1300':
        metamers = body_resps[:,:1000,:]
    else:
        metamers = body_resps[:,300:1300,:]
    # generate concated response.
    if i == 0:
        all_matemer_resp = copy.deepcopy(metamers)
    else:
        all_matemer_resp = np.concatenate([all_matemer_resp,metamers],axis=0)
    # generate cell infos.
    c_dps = a.Cell_FOB_DPrimes
    c_rsp = a.Cell_FOB_Response_avr
    c_dps = c_dps[c_dps['Cell'].isin(body_cells)].copy() # only body-selected part
    c_rsp = c_rsp[c_rsp['Cell'].isin(body_cells)].copy()

    # add loc.
    c_dps.loc[:,'Loc'] = c_loc
    c_rsp.loc[:,'Loc'] = c_loc
    if i ==0:
        all_d_primes = copy.deepcopy(c_dps)
        all_response = copy.deepcopy(c_rsp)
    else:
        all_d_primes = pd.concat((all_d_primes,c_dps))
        all_response = pd.concat((all_response,c_rsp))

all_d_primes['Cell_ID'] = all_d_primes.groupby(['Loc', 'Cell'], sort=False).ngroup()
all_response['Cell_ID'] = all_response.groupby(['Loc', 'Cell'], sort=False).ngroup()

#%% save all msb cells
np.savez_compressed(ot.Join(save_path,'AL_Cells_Metamer_Only.npz'),psth = all_matemer_resp,d_primes = all_d_primes,response = all_response)



