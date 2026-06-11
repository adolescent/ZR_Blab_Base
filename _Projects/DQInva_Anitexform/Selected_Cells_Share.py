#%% paths

DQInva_site_path = r'E:\#Preprocessed_Data\SiteClass\DQInva'
save_path = r'E:\#Preprocessed_Data\Selected_Cells\_DQInva_For_Share\Raw_Data\Selected_Cells'

ceiling, dp_thres = 0.2, 0.5  # split-half noise ceiling, FOB d-prime


#%% collect MSB(body) + ML(face) cells per site -> compressed npz (4D uint8)

import os
import joblib as JL
import numpy as np
import OS_Tools as ot

_SUBJ_MAP = {
    'Maodan': 'MonkeyM', 'MaoDan': 'MonkeyM', 'MonkeyM': 'MonkeyM',
    'JianJian': 'MonkeyJ', 'JJ': 'MonkeyJ', 'MonkeyJ': 'MonkeyJ',
}


def share_site_key(site_name):
    date, subj = site_name.split('_')[0], site_name.split('_')[1]
    return f'{date}_{_SUBJ_MAP.get(subj, subj)}'


site_data = {}        # site_key -> (N_Cell, N_Repeat, N_Img, N_Time) uint8
cell_site = []        # owner site_key for each kept neuron
cell_type = []        # 'MSB' (body), 'ML' (face) or 'MSB+ML' (both)
cell_local_idx = []   # original cell index inside its own site

for site in sorted(ot.Get_File_Name(DQInva_site_path, '.joblib')):
    a = JL.load(site)
    body_cells, _ = a.Cell_Selection(ceiling=ceiling, prefer='body', dp_thres=dp_thres)
    face_cells, _ = a.Cell_Selection(ceiling=ceiling, prefer='face', dp_thres=dp_thres)
    cells = np.union1d(body_cells, face_cells)
    if len(cells) == 0:
        del a
        continue

    key = share_site_key(a.site_name)
    site_data[key] = np.ascontiguousarray(a.raw_psth[cells], dtype=np.uint8)

    body_set, face_set = set(body_cells.tolist()), set(face_cells.tolist())
    for c in cells.tolist():
        in_b, in_f = c in body_set, c in face_set
        cell_type.append('MSB+ML' if in_b and in_f else ('MSB' if in_b else 'ML'))
        cell_site.append(key)
        cell_local_idx.append(c)
    print(f'{a.site_name} -> {key}: {len(cells)} cells, matrix {site_data[key].shape}')
    del a

os.makedirs(save_path, exist_ok=True)
out_path = ot.Join(save_path, 'DQInva_selected_cells_raw_psth.npz')
np.savez_compressed(
    out_path,
    cell_site=np.array(cell_site),
    cell_type=np.array(cell_type),
    cell_local_idx=np.array(cell_local_idx, dtype=np.int32),
    **site_data,
)
print(f'saved {out_path}: {len(cell_site)} cells from {len(site_data)} sites')


#%% usage example
# z = np.load(out_path, allow_pickle=False)
# z['cell_site']                 # site key for each neuron, e.g. '260509_MonkeyM'
# z['cell_type']                 # 'MSB' / 'ML' / 'MSB+ML'
# mat = z['260509_MonkeyM']      # (N_Cell, N_Repeat, N_Img, N_Time), rows align with cell_site==key
