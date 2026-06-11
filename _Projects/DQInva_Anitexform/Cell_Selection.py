
#%%

import os
import OS_Tools as ot
import joblib as JL
import numpy as np

DQInva_site_path = r'E:\#Preprocessed_Data\SiteClass\DQInva'
DQInva_savepath = r'E:\#Preprocessed_Data\Selected_Cells\DQInva_Stable'

ceiling, dp_thres = 0.2, 0.5
t_win = slice(150, 420)
psth_n_time = 450
n_fold, n_pseudo = 5, 10


def _pair_avg(x):
    """(cell, trial, 2*n_img) -> (cell, trial, n_img), average duplicate stim blocks."""
    return x.reshape(x.shape[0], x.shape[1], 2, -1).mean(2)


def _avg_repeat(x, n_repeat=2):
    return x.reshape(x.shape[0], n_repeat, -1, *x.shape[2:]).mean(1)


def _pseudo_folds(rsp, n_fold=n_fold, n_pseudo=n_pseudo, seed=0):
    """rsp: (cell, n_repeat, n_img). Split-half CV; train/test trials do not overlap."""
    n_cell, n_repeat, n_img = rsp.shape
    half = n_repeat // 2
    if half < 1:
        raise ValueError(f'need >=2 trial repeats, got {n_repeat}')
    rng = np.random.default_rng(seed)
    folds = []
    for _ in range(n_fold):
        perm = rng.permutation(n_repeat)
        tr, te = perm[:half], perm[half:]
        test = rsp[:, te, :].mean(1)
        train = np.stack([
            rsp[:, rng.choice(tr, size=n_img), np.arange(n_img)]
            for _ in range(n_pseudo)
        ])
        folds.append({
            'train': train, 'test': test,
            'train_trials': tr, 'test_trials': te, 'n_repeat': n_repeat,
        })
    return folds


def _merge_folds(site_folds, site_names):
    merged = []
    for f in range(len(site_folds[0])):
        merged.append({
            'train': np.concatenate([sf[f]['train'] for sf in site_folds], axis=1),
            'test': np.vstack([sf[f]['test'] for sf in site_folds]),
            'by_site': {
                name: {
                    'train_trials': site_folds[i][f]['train_trials'],
                    'test_trials': site_folds[i][f]['test_trials'],
                    'n_repeat': site_folds[i][f]['n_repeat'],
                }
                for i, name in enumerate(site_names)
            },
        })
    return merged


def _extract(site, prefer):
    a = JL.load(site)
    stim_info = a.stim_info
    cells, psth = a.Cell_Selection(ceiling=ceiling, prefer=prefer, dp_thres=dp_thres)
    psth = psth[..., :psth_n_time]
    redplot = psth[:, :, t_win].sum(-1)

    msk = (stim_info.Stim_Set == 'ShadingTex1') & (stim_info.Object != 0)
    msk = msk.to_numpy()
    set1_rsp = _avg_repeat(redplot[:, msk])
    set2_rsp = _avg_repeat(redplot[:, 360:])
    set1_psth = _avg_repeat(psth[:, msk, :])
    set2_psth = _avg_repeat(psth[:, 360:, :])
    fob_msk = stim_info['Stim_Set'].str.contains('FOB', na=False).to_numpy()
    fob_rsp = a.raw_redplot[cells][:, fob_msk]

    trial_rsp = a.raw_psth[cells][:, :, :, t_win].sum(-1)
    set1_trial = _pair_avg(trial_rsp[:, :, msk])
    set2_trial = _pair_avg(trial_rsp[:, :, 360:])
    return (set1_rsp, set2_rsp, set1_psth, set2_psth, fob_rsp, cells,
            set1_trial, set2_trial, a.site_name)


def _zscore(x):
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


#%% select MSB (body) and ML (face) cells, pool both recordings

for tag, prefer in [('MSB', 'body'), ('ML', 'face')]:
    s1, s2, p1, p2, fob = [], [], [], [], []
    s1_fold_sites, s2_fold_sites, site_names = [], [], []
    cell_site_idx, cell_local_idx = [], []

    for site_i, site in enumerate(ot.Get_File_Name(DQInva_site_path, '.joblib')):
        (r1, r2, t1, t2, f, c, tr1, tr2, site_name) = _extract(site, prefer)
        s1.append(r1); s2.append(r2); p1.append(t1); p2.append(t2); fob.append(f)
        site_names.append(site_name)
        cell_site_idx.extend([site_i] * len(c))
        cell_local_idx.extend(c.tolist())
        s1_fold_sites.append(_pseudo_folds(tr1, seed=site_i * 2))
        s2_fold_sites.append(_pseudo_folds(tr2, seed=site_i * 2 + 1))

    out_dir = ot.Join(DQInva_savepath, tag)
    os.makedirs(out_dir, exist_ok=True)
    np.save(ot.Join(out_dir, 'set1_rsp_z.npy'), _zscore(np.vstack(s1)))
    np.save(ot.Join(out_dir, 'set2_rsp_z.npy'), _zscore(np.vstack(s2)))
    np.save(ot.Join(out_dir, 'set1_psth.npy'), np.vstack(p1))
    np.save(ot.Join(out_dir, 'set2_psth.npy'), np.vstack(p2))
    np.save(ot.Join(out_dir, 'fob_rsp.npy'), np.vstack(fob))
    np.save(ot.Join(out_dir, 'psth_time_ms.npy'), np.arange(psth_n_time))

    pseudo = {
        'prefer': prefer,
        'n_fold': n_fold,
        'n_pseudo': n_pseudo,
        'site_names': site_names,
        'cell_site_idx': np.array(cell_site_idx, np.int32),
        'cell_local_idx': np.array(cell_local_idx, np.int32),
        'set1': {
            'n_img': s1_fold_sites[0][0]['train'].shape[2],
            'folds': _merge_folds(s1_fold_sites, site_names),
        },
        'set2': {
            'n_img': s2_fold_sites[0][0]['train'].shape[2],
            'folds': _merge_folds(s2_fold_sites, site_names),
        },
    }
    JL.dump(pseudo, ot.Join(out_dir, 'pseudo_trials.joblib'), compress=3)

    print(f'{tag}: {len(cell_site_idx)} cells from {len(site_names)} recordings')
    print(f'  set1 {pseudo["set1"]["n_img"]} imgs, set2 {pseudo["set2"]["n_img"]} imgs, '
          f'{n_fold} folds x {n_pseudo} train pseudos')

#%% usage example
# data = JL.load(ot.Join(DQInva_savepath, 'msb', 'pseudo_trials.joblib'))
# fold0 = data['set1']['folds'][0]
# X_train = fold0['train']          # (10, N_cell, N_img)
# X_test  = fold0['test']           # (N_cell, N_img)
# fold0['by_site'][data['site_names'][0]]['train_trials']  # trial idx used for train pool
