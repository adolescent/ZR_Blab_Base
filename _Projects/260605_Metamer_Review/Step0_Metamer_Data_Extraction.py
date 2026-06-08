'''
Get all metamer result from all recording sites.

We need:
1. All FOB Response
2. All 1k metamer response(avr)
3. All 1k metamer response(by psth)
4. Save all trail data in dict file, for usage of decoding&encoding.


For all 4 brain areas(ML,MSB,ASB,AL)

'''


#%%

import joblib as JL
import OS_Tools as ot
from Py_Structure.Cell_Selector import Stim_Cell_Rearrange
from Py_Structure.Info_Files.InfoLoader import Select_Cell_Info
import pandas as pd
import numpy as np
from Py_Structure.Struct_Funcs import Single_Recording_Site
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


root_path = r'E:\#Preprocessed_Data\SiteClass\Metamers'
brain_areas = ['ML','MSB','ASB','AL']

savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
N_FOB = 150
N_IMG = 1000


def _pad_fob(fob, n_fob=N_FOB):
    """FOB72/STI150 -> (N_cell, N_FOB); unused cols are NaN."""
    fob = np.asarray(fob, dtype=np.float32)
    if fob.shape[1] >= n_fob:
        return fob[:, :n_fob], fob.shape[1]
    out = np.full((fob.shape[0], n_fob), np.nan, dtype=np.float32)
    out[:, :fob.shape[1]] = fob
    return out, fob.shape[1]


def _match_brain_area(SRS, target_area):
    """ML/MF share MSB folder; MSB/ASB/AL filter by SRS.brain_areas."""
    areas = getattr(SRS, 'brain_areas', [])
    if target_area == 'ML':
        return 'ML' in areas or 'MF' in areas
    return target_area in areas


#%%
for cloc in brain_areas:
    if cloc in ('ML', 'MSB'):
        data_path = ot.Join(root_path, 'MSB')
    else:
        data_path = ot.Join(root_path, 'AL_ASB')

    ceiling_thres = 0.3
    dp_thres = 0.5
    prefer = 'Face' if cloc in ('ML', 'AL') else 'Body'

    avr_list, psth_list, fob_list = [], [], []
    fob_valid_len, fob_style = [], []
    site_names, cell_site_idx, cell_local_idx = [], [], []
    by_site = {}
    stim_infos = Select_Cell_Info('Metamer_1k')

    sites = ot.Get_File_Name(data_path, '.joblib')
    for c_site in tqdm(sites, total=len(sites), desc=cloc):
        SRS = JL.load(c_site)
        if not _match_brain_area(SRS, cloc):
            del SRS
            continue
        site_key = SRS.site_name
        c_fob, c_avr, c_psth, c_rsp_by_trail, c_psth_by_trail, _ = Stim_Cell_Rearrange(
            Cell_Class=SRS,
            ceiling_thres=ceiling_thres,
            prefer=prefer,
            dp_thres=dp_thres,
            stim_type='Metamer_1k',
            time_start=160,
            time_end=320,
        )
        if c_avr.shape[0] == 0:
            continue

        fob_style_name = stim_infos[SRS.stimset]['FOB']['style']

        c_fob, n_fob_valid = _pad_fob(c_fob)
        n_cell = c_avr.shape[0]
        site_idx = len(site_names)
        site_names.append(site_key)

        avr_list.append(c_avr)
        psth_list.append(c_psth)
        fob_list.append(c_fob)
        fob_valid_len.extend([n_fob_valid] * n_cell)
        fob_style.extend([fob_style_name] * n_cell)
        cell_site_idx.extend([site_idx] * n_cell)
        cell_local_idx.extend(np.arange(n_cell))

        by_site[site_key] = {
            'rsp_by_trial': c_rsp_by_trail,
            'psth_by_trial': c_psth_by_trail,
            'n_repeat': c_rsp_by_trail.shape[1],
            'fob_style': fob_style_name,
            'stimset': SRS.stimset,
        }
        del SRS

    if not avr_list:
        print(f'{cloc}: no cells selected, skip save.')
        continue

    out = {
        # pooled point data (2-3): row i = global cell i
        'avr': np.vstack(avr_list),                          # (N_cell, 1000)
        'psth': np.vstack(psth_list),                        # (N_cell, 1000, 450)
        'fob': np.vstack(fob_list),                          # (N_cell, 150), NaN-padded
        'fob_valid_len': np.array(fob_valid_len, np.int16),  # 72 or 150 per cell
        'fob_style': np.array(fob_style),                    # 'FOB72' / 'STI150'
        # neuron -> recording site
        'site_names': site_names,                            # list[str], site index -> name
        'cell_site_idx': np.array(cell_site_idx, np.int32),  # global cell -> site index
        'cell_local_idx': np.array(cell_local_idx, np.int32),  # global cell -> index within site
        # per-site trial data (4): N_repeat differs across sites
        'by_site': by_site,
        'brain_area': cloc,
        'prefer': prefer,
    }
    ot.Mkdir(savepath)
    JL.dump(out, ot.Join(savepath, f'{cloc}_Metamer_1k.joblib'), compress=3)


#%%  usage example
# data = JL.load(ot.Join(savepath, 'ML_Metamer_1k.joblib'))
# i = 0
# site = data['site_names'][data['cell_site_idx'][i]]
# local_i = data['cell_local_idx'][i]
# trial_rsp = data['by_site'][site]['rsp_by_trial'][local_i]       # (N_repeat, 1000)
# trial_psth = data['by_site'][site]['psth_by_trial'][local_i]     # (N_repeat, 1000, 450)


