'''
Select NSD 1k noise ceiled response.
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
from tqdm import tqdm
warnings.filterwarnings("ignore")

# msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\Metamers\AL_ASB','.joblib')
save_path = r'E:\#Preprocessed_Data\Selected_Cells'
target_area = 'MSB'


msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\nsd','.joblib')

#%%
FOB_PLACEHOLDER = np.array(None, dtype=object)  # NSD has no FOB
DF_PLACEHOLDER = np.array(None, dtype=object)   # placeholder for d_primes/response

all_nsd_resp_list = []
all_ceiling_index_list = []
all_cell_info_list = []

ceiling_threshold = 0.3

for _, cloc in tqdm(enumerate(msb_sites), total=len(msb_sites)):
    fname = cloc.split('\\')[-1]
    print(fname)
    try:
        a = JL.load(cloc)
    except Exception as e:
        print(f'Failed to load {fname}: {e}')
        continue

    # select brain area
    if target_area is not None and hasattr(a, 'brain_areas'):
        if target_area not in a.brain_areas:
            print(f'Not Correct Area, ignore {fname}.')
            del a
            gc.collect()
            continue

    # recompute noise ceiling using updated API, then persist to Site_Info
    try:
        if not hasattr(a, 'raw_psth'):
            raise AttributeError('raw_psth not found on loaded site object.')
        if not hasattr(a, 'used_on'):
            a.used_on = np.arange(160, 320)

        a.Noise_Ceiling(method='all')

        if hasattr(a, 'Site_Info') and isinstance(a.Site_Info, pd.DataFrame):
            if 'Ceiling_Index' in a.Site_Info.columns:
                a.Site_Info.loc[:, 'Ceiling_Index'] = a.ceiling_index
            else:
                a.Site_Info['Ceiling_Index'] = a.ceiling_index
        else:
            a.Site_Info = pd.DataFrame({'Cell': np.arange(len(a.ceiling_index)), 'Ceiling_Index': a.ceiling_index})
    except Exception as e:
        print(f'Noise ceiling recompute failed for {fname}: {e}')
        del a
        gc.collect()
        continue

    # resave updated object to the same path
    try:
        JL.dump(a, cloc, compress=7)
    except Exception as e:
        print(f'Failed to resave {fname}: {e}')

    # select ok cells by ceiling index only (no correction)
    try:
        ok_cells = np.array(a.Site_Info[a.Site_Info.Ceiling_Index > ceiling_threshold].Cell).astype('i4')
    except Exception:
        ok_cells = np.where(np.asarray(a.ceiling_index) > ceiling_threshold)[0].astype('i4')

    if ok_cells.size == 0:
        del a
        gc.collect()
        continue

    if not hasattr(a, 'avr_psth'):
        print(f'avr_psth not found, ignore {fname}.')
        del a
        gc.collect()
        continue

    selected_resps = a.avr_psth[ok_cells, :, :]
    all_nsd_resp_list.append(selected_resps)
    all_ceiling_index_list.append(np.asarray(a.ceiling_index)[ok_cells])

    c_info = a.Site_Info.copy() if hasattr(a, 'Site_Info') else pd.DataFrame({'Cell': ok_cells})
    if 'Cell' not in c_info.columns:
        c_info['Cell'] = np.arange(len(c_info))
    c_info = c_info[c_info['Cell'].isin(ok_cells)].copy()

    c_loc = getattr(a, 'site_name', fname.replace('.joblib', ''))
    c_info['Loc'] = c_loc
    c_info['Stimset'] = getattr(a, 'stimset', 'NSD_1k')
    all_cell_info_list.append(c_info.reset_index(drop=True))

    del a
    gc.collect()

all_nsd_resp = np.concatenate(all_nsd_resp_list, axis=0) if len(all_nsd_resp_list) else np.empty((0, 0, 0))
all_ceiling_index = np.concatenate(all_ceiling_index_list, axis=0) if len(all_ceiling_index_list) else np.empty((0,))
all_cell_info = pd.concat(all_cell_info_list, ignore_index=True) if len(all_cell_info_list) else pd.DataFrame()

if len(all_cell_info):
    all_cell_info['Cell_ID'] = all_cell_info.groupby(['Loc', 'Cell'], sort=False).ngroup()

#%% save all nsd ok cells
out_path = ot.Join(save_path, f'{target_area}_Cells_NSD_1k_Ceiling_{ceiling_threshold:.2f}.npz')
np.savez_compressed(
    out_path,
    psth=all_nsd_resp,
    FOB=FOB_PLACEHOLDER,
    d_primes=DF_PLACEHOLDER,
    response=DF_PLACEHOLDER,
    ceiling_index=all_ceiling_index,
    cell_info=all_cell_info
)

print(f'Saved: {out_path}')