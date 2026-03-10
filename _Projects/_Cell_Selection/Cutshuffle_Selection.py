'''
This script will select good body cells in MSB.

NOTE only body cells, and only cut-shuffle. 

'''


#%%

from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
import copy
import warnings
import gc
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB','.joblib')
save_path = r'E:\#Preprocessed_Data\Selected_Cells'
target_area = 'ASB'


if (target_area == 'ASB') or (target_area == 'AL'):
    msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB','.joblib')
else:
    msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\Metamers\MSB','.joblib')



#%%
def _get_stimset_series(stim_info):
    """
    Return a pandas Series of Stim_Set labels if available; otherwise None.
    Handles both DataFrame and objects with attribute access.
    """
    if stim_info is None:
        return None
    if isinstance(stim_info, pd.DataFrame) and ('Stim_Set' in stim_info.columns):
        return stim_info['Stim_Set']
    if hasattr(stim_info, 'Stim_Set'):
        return pd.Series(getattr(stim_info, 'Stim_Set'))
    return None


all_metamer_resp_list = []
all_fob_resp_list = []
all_d_primes_list = []
all_response_list = []
expected_fob_n = 300

for _, cloc in tqdm(enumerate(msb_sites), total=len(msb_sites)):
    fname = cloc.split('\\')[-1]
    print(fname)
    a = JL.load(cloc)
    c_info = a.Site_Info
    c_loc = a.site_name

    # select brain area
    if target_area == 'ML':
        if ('ML' not in a.brain_areas) and ('MF' not in a.brain_areas):
            print(f'Not Correct Area, ignore {fname}.')
            del a
            gc.collect()
            continue
    elif target_area not in a.brain_areas:
        print(f'Not Correct Area, ignore {fname}.')
        del a
        gc.collect()
        continue

    ok_cells = np.array(c_info[c_info.Ceiling_Index>(0.3/1.7)].Cell)
    all_cell_dps = a.Cell_FOB_DPrimes.pivot(index='Cell',columns='Category',values='D_Prime')
    if target_area == 'MSB' or target_area == 'ASB':
        tuned_cells = np.array(all_cell_dps[all_cell_dps['Body']>0.5].index)
    else:
        tuned_cells = np.array(all_cell_dps[all_cell_dps['Face']>0.5].index)
    selected_cells = np.intersect1d(ok_cells, tuned_cells)
    selected_cells = selected_cells.astype('i4') 
    if selected_cells.size == 0:
        del a
        gc.collect()
        continue

    selected_resps = a.avr_psth[selected_cells, :, :]

    # Metamer subset (kept as your original logic)
        # raw_rsp = a.avr_psth
    if a.stimset == 'Mega_Metamer_v250920':
        metamers = selected_resps[:,300:,:]
    elif a.stimset == 'Mega_Metamer_v251104':
        metamers = selected_resps[:,300:2240,:]
    else:
        print(f'{c_loc} not correct stimset, ignore.')
        del a
        gc.collect()
        continue

    all_metamer_resp_list.append(metamers)

    # FOB subset: indices where stim_info.Stim_Set contains 'FOB'
    stim_set = _get_stimset_series(getattr(a, 'stim_info', None))
    if stim_set is None:
        fob_resp = selected_resps[:, :0, :]  # empty but concat-safe if consistent
    else:
        fob_sets = np.where(stim_set.astype(str).str.contains('FOB', na=False).to_numpy())[0]
        fob_resp = selected_resps[:, fob_sets, :]

    if expected_fob_n is None and fob_resp.shape[1] > 0:
        expected_fob_n = fob_resp.shape[1]
    elif expected_fob_n is not None and fob_resp.shape[1] != expected_fob_n:
        # Keep concatenation robust across sites with slightly different FOB counts
        print('Warning of FOB: Not STI 150*2. Fill with 300.')
        min_n = min(expected_fob_n, fob_resp.shape[1])
        fob_resp = fob_resp[:, :min_n, :]
        # expected_fob_n = min_n
    all_fob_resp_list.append(fob_resp)

    # cell info tables
    c_dps = a.Cell_FOB_DPrimes
    c_rsp = a.Cell_FOB_Response_avr
    c_dps = c_dps[c_dps['Cell'].isin(selected_cells)].copy()
    c_rsp = c_rsp[c_rsp['Cell'].isin(selected_cells)].copy()

    # add loc and stimset (same as Doodle_Acquire.py)
    c_dps['Loc'] = c_loc
    c_rsp['Loc'] = c_loc
    c_dps['Stimset'] = a.stimset
    c_rsp['Stimset'] = a.stimset

    all_d_primes_list.append(c_dps)
    all_response_list.append(c_rsp)

    del a
    gc.collect()

all_matemer_resp = np.concatenate(all_metamer_resp_list, axis=0) if len(all_metamer_resp_list) else np.empty((0, 0, 0))
# Ensure that all fob_resp arrays have shape (N_cells, 300, 450); pad with NaN if only 72 FOBs are present
normalized_fob_list = []
for arr in all_fob_resp_list:
    if arr.shape[1] == 72:
        pad_width = ((0, 0), (0, 300 - 72), (0, 0))
        arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    normalized_fob_list.append(arr)
all_fob_resp = np.concatenate(normalized_fob_list, axis=0) if len(normalized_fob_list) else np.empty((0, 0, 0))
all_d_primes = pd.concat(all_d_primes_list, ignore_index=True) if len(all_d_primes_list) else pd.DataFrame()
all_response = pd.concat(all_response_list, ignore_index=True) if len(all_response_list) else pd.DataFrame()

if len(all_d_primes):
    all_d_primes['Cell_ID'] = all_d_primes.groupby(['Loc', 'Cell'], sort=False).ngroup()
if len(all_response):
    all_response['Cell_ID'] = all_response.groupby(['Loc', 'Cell'], sort=False).ngroup()
#%% ## Plot response heatmap.
fob_avr = np.nan_to_num(all_fob_resp[:,:,160:320]).sum(-1)
plotable = fob_avr/fob_avr.max(1,keepdims = True)
sns.heatmap(plotable,center=0,cmap='bwr')


#%% save all msb cells
np.savez_compressed(
    ot.Join(save_path, f'{target_area}_Cells_Metamer_Cutshuffle.npz'),
    psth=all_matemer_resp,
    FOB=all_fob_resp,
    d_primes=all_d_primes,
    response=all_response
)

print(set(all_d_primes.Loc))
#%% counts 
counter = 0
acs = all_d_primes[all_d_primes.Category=='Face'].reset_index(drop=True)
for i in range(len(acs)):
    cc_loc = acs.loc[i,'Loc']
    # if ('MD' in cc_loc) or ('Mao' in cc_loc):
    if 'Zhuang' in cc_loc:
        counter+=1
print(len(acs))
print(counter)




#%%


