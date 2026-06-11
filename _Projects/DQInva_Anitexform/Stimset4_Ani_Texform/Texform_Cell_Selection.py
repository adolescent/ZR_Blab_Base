
#%%

import os
import OS_Tools as ot
import joblib as JL
import numpy as np

sitepath = r'E:\#Preprocessed_Data\SiteClass\ani_texform'
savepath = r'E:\#Preprocessed_Data\Selected_Cells\Ani_Texform'

ceiling, dp_thres = 0.2, 0.5
n_fob = 72
texform_start = 72
t_win = slice(150, 320)


def _zscore(x):
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


def _extract(site, prefer):
    a = JL.load(site)
    cells, psth = a.Cell_Selection(ceiling=ceiling, prefer=prefer, dp_thres=dp_thres)
    texform_psth = psth[:, texform_start:, :]
    redplot = texform_psth[:, :, t_win].sum(-1)
    fob_rsp = a.raw_redplot[cells][:, :n_fob]
    trial_psth = a.raw_psth[cells][:, :, texform_start:, :]
    return texform_psth, trial_psth, redplot, fob_rsp, cells, trial_psth.shape[1], a.site_name


#%% select MSB (body) and ML (face) cells, pool both recordings

for tag, prefer in [('MSB', 'body'), ('ML', 'face')]:
    avr_list, trial_list, red_list, fob_list = [], [], [], []
    site_names, cell_site_idx, cell_local_idx = [], [], []
    n_trials_by_site = []

    for site_i, site in enumerate(ot.Get_File_Name(sitepath, '.joblib')):
        avr_psth, trial_psth, redplot, fob_rsp, cells, n_trial, site_name = _extract(site, prefer)
        avr_list.append(avr_psth)
        trial_list.append(trial_psth)
        red_list.append(redplot)
        fob_list.append(fob_rsp)
        site_names.append(site_name)
        n_trials_by_site.append(n_trial)
        cell_site_idx.extend([site_i] * len(cells))
        cell_local_idx.extend(cells.tolist())

    n_trials_pooled = min(n_trials_by_site)
    trial_pooled = [x[:, :n_trials_pooled] for x in trial_list]

    out_dir = ot.Join(savepath, tag)
    os.makedirs(out_dir, exist_ok=True)
    avr_all = np.vstack(avr_list)
    np.save(ot.Join(out_dir, 'avr_psth.npy'), avr_all)
    np.save(ot.Join(out_dir, 'by_trial_psth.npy'), np.vstack(trial_pooled))
    np.save(ot.Join(out_dir, 'avr_rsp_z.npy'), _zscore(np.vstack(red_list)))
    np.save(ot.Join(out_dir, 'fob_rsp.npy'), np.vstack(fob_list))
    np.save(ot.Join(out_dir, 'psth_time_ms.npy'), np.arange(avr_all.shape[-1]))

    meta = {
        'prefer': prefer,
        'site_names': site_names,
        'cell_site_idx': np.array(cell_site_idx, np.int32),
        'cell_local_idx': np.array(cell_local_idx, np.int32),
        'n_trials_by_site': np.array(n_trials_by_site, np.int32),
        'n_trials_pooled': n_trials_pooled,
        'n_fob': n_fob,
        'texform_start': texform_start,
        'n_texform': avr_all.shape[1],
        't_win': (t_win.start, t_win.stop),
    }
    JL.dump(meta, ot.Join(out_dir, 'meta.joblib'), compress=3)

    print(f'{tag}: {len(cell_site_idx)} cells from {len(site_names)} recordings')
    print(f'  avr_psth {avr_all.shape}, by_trial {np.vstack(trial_pooled).shape}, '
          f'trials/site {n_trials_by_site} -> pooled {n_trials_pooled}')

#%% usage example
# meta = JL.load(ot.Join(savepath, 'MSB', 'meta.joblib'))
# avr_psth = np.load(ot.Join(savepath, 'MSB', 'avr_psth.npy'))
# by_trial = np.load(ot.Join(savepath, 'MSB', 'by_trial_psth.npy'))  # (cell, trial, stim, time)
