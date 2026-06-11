#%% paths

share_root = r'E:\#Preprocessed_Data\Selected_Cells\_DQInva_For_Share'
raw_path = r'E:\#Preprocessed_Data\Selected_Cells\_DQInva_For_Share\Raw_Response'
stable_path = r'E:\#Preprocessed_Data\SiteClass\DQInva'
video_path = r'E:\#Preprocessed_Data\SiteClass\DQInva_video'


#%% export raw_psth from joblib -> compressed npz (4D uint8 per site)

import os
import joblib as JL
import seaborn as sns
import numpy as np
import OS_Tools as ot

_SUBJ_MAP = {
    'Maodan': 'MonkeyM', 'MaoDan': 'MonkeyM', 'MonkeyM': 'MonkeyM',
    'JianJian': 'MonkeyJ', 'JJ': 'MonkeyJ', 'MonkeyJ': 'MonkeyJ',
}


def share_site_key(site_name):
    date, subj = site_name.split('_')[0], site_name.split('_')[1]
    return f'{date}_{_SUBJ_MAP.get(subj, subj)}'


def export_raw_psth(joblib_dir, out_name):
    site_data = {}
    for site_path in sorted(ot.Get_File_Name(joblib_dir, '.joblib')):
        a = JL.load(site_path)
        key = share_site_key(a.site_name)
        site_data[key] = np.ascontiguousarray(a.raw_psth, dtype=np.uint8)
        print(f'{a.site_name} -> {key}, shape {site_data[key].shape}')
        del a

    os.makedirs(raw_path, exist_ok=True)
    out_path = ot.Join(raw_path, out_name)
    np.savez_compressed(out_path, **site_data)
    print(f'saved {out_path}, keys: {list(site_data.keys())}')


#%% stable DQInva
export_raw_psth(stable_path, 'DQInva_stable_raw_psth.npz')

#%% video DQInva
export_raw_psth(video_path, 'DQInva_video_raw_psth.npz')

#%% test

a = np.load(ot.Join(raw_path, 'DQInva_video_raw_psth.npz'))
t = a['260509_MonkeyM']
rsp = t[:, :, :, 1050:1320].sum((1, 3))
sns.heatmap(rsp / rsp.max(1, keepdims=True), vmax=1, vmin=0, cmap='Reds')

#%% select MSB / ML cells -> psth.npz + rsp.npz per tag

ceiling, dp_thres = 0.2, 0.5
_onset_stable = 100
_t_stable = slice(_onset_stable + 50, _onset_stable + 320)
_onset_video = 1000
_t_video = slice(_onset_video, _onset_video + 1000)
_n_fob = 72
_n_cycle = 3


def _pivot_repeat(x, n_rep=2):
    """(cell, n_trial, n_rep*n_stim, time) -> (cell, n_trial*n_rep, n_stim, time)."""
    c, tr, ns = x.shape[0], x.shape[1], x.shape[2] // n_rep
    return x.reshape(c, tr, n_rep, ns, x.shape[3]).reshape(c, tr * n_rep, ns, x.shape[3])


def export_selected_cells(tag, prefer):
    s1p, s2p, s1r, s2r, site = [], [], [], [], []
    for p in sorted(ot.Get_File_Name(stable_path, '.joblib')):
        a = JL.load(p)
        cells, _ = a.Cell_Selection(ceiling=ceiling, prefer=prefer, dp_thres=dp_thres)
        if len(cells) == 0:
            continue
        raw = np.ascontiguousarray(a.raw_psth[cells], dtype=np.uint8)
        msk = ((a.stim_info.Stim_Set == 'ShadingTex1') & (a.stim_info.Object != 0)).to_numpy()
        key = share_site_key(a.site_name)
        p1 = _pivot_repeat(raw[:, :, msk, :])
        p2 = _pivot_repeat(raw[:, :, 360:, :])
        s1p.append(p1); s2p.append(p2)
        s1r.append(p1[:, :, :, _t_stable].sum(-1, dtype=np.uint16))
        s2r.append(p2[:, :, :, _t_stable].sum(-1, dtype=np.uint16))
        site.extend([key] * len(cells))
        print(f'stable {a.site_name}: {len(cells)} cells, {p1.shape}')
        del a

    vp, vr, vsite = [], [], []
    for p in sorted(ot.Get_File_Name(video_path, '.joblib')):
        a = JL.load(p)
        cells, _ = a.Cell_Selection(ceiling=ceiling, prefer=prefer, dp_thres=dp_thres)
        if len(cells) == 0:
            continue
        psth = np.ascontiguousarray(a.raw_psth[cells][:, :, _n_fob:, :], dtype=np.uint8)
        key = share_site_key(a.site_name)
        psth = _pivot_repeat(psth, n_rep=_n_cycle)
        vp.append(psth)
        vr.append(psth[:, :, :, _t_video].sum(-1, dtype=np.uint16))
        vsite.extend([key] * len(cells))
        print(f'video {a.site_name}: {len(cells)} cells, {psth.shape}')
        del a

    os.makedirs(share_root, exist_ok=True)
    cell_site = np.array(site)
    video_cell_site = np.array(vsite)
    base = ot.Join(share_root, f'DQInva_{tag}')
    np.savez_compressed(f'{base}_psth.npz',
        set1_psth=np.concatenate(s1p), set2_psth=np.concatenate(s2p),
        video_psth=np.concatenate(vp), cell_site=cell_site, video_cell_site=video_cell_site)
    np.savez_compressed(f'{base}_rsp.npz',
        set1_rsp=np.concatenate(s1r), set2_rsp=np.concatenate(s2r),
        video_rsp=np.concatenate(vr), cell_site=cell_site, video_cell_site=video_cell_site)
    print(f'saved {base}_psth.npz / {base}_rsp.npz')


export_selected_cells('MSB', 'body')
export_selected_cells('ML', 'face')


#%% usage example
p = np.load(ot.Join(share_root, 'DQInva_MSB_psth.npz'))
r = np.load(ot.Join(share_root, 'DQInva_MSB_rsp.npz'))
# p['set1_psth']   # (N_cell, 28, N_stim, N_time)
# r['set2_rsp']    # (N_cell, 28, N_stim), sum over psth[:,:,:,150:420]

#%% npz -> mat (v5; root 4 selected files only)

from scipy.io import savemat

_MAT_V5_LIMIT = 2 * 1024 ** 3
_npz_list = [
    'DQInva_MSB_psth.npz', 'DQInva_MSB_rsp.npz',
    'DQInva_ML_psth.npz', 'DQInva_ML_rsp.npz',
]

for name in _npz_list:
    npz_path = ot.Join(share_root, name)
    if not os.path.exists(npz_path):
        print(f'skip (not found): {npz_path}')
        continue
    z = np.load(npz_path, allow_pickle=True)
    mdict, nbytes = {}, 0
    for k in z.files:
        v = z[k]
        mdict[k] = np.array([str(x) for x in v], dtype=object) if v.dtype.kind in 'SU' else v
        nbytes += v.nbytes
        if v.nbytes > _MAT_V5_LIMIT:
            print(f'WARNING: {name} [{k}] {v.nbytes / 1e9:.2f} GB > 2 GB mat-v5 limit')
    out = npz_path.replace('.npz', '.mat')
    savemat(out, mdict, do_compression=True, format='5')
    print(f'{name} -> {out}  ({nbytes / 1e6:.1f} MB uncompressed)')
#%%

