'''
对 MSB / ML / ASB / AL 四个脑区批量运行 50D 样本空间分析（Bao et al. 方法）。

共享结果保存在 savepath 根目录（object space、shuffle 轴）；
各脑区结果保存在 savepath/{area}/ 下。

运行: python All_Brain_Areas.py
'''

#%% 配置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

nsd_figpath = r'E:\#Stimsets\NSD1000'
metamer_figpath = r'E:\#Stimsets\Metamer_P4_C4321_Object_STI150_1300'
cell_rootpath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis'

N_DIM = 50
N_METAMER = 1000
N_OBJ = 40
N_SHUF = 5
BATCH = 32
N_EXTREME = 10

#%% 依赖
import sys
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / 'Common_Functions'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from obj_space_paths import BRAIN_AREAS, area_dir, area_path, rsp_path, shared_path

import OS_Tools as ot
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as T
from sklearn.decomposition import PCA
from PIL import Image


def _ensure_dirs():
    ot.Mkdir(savepath, mute=True)
    for area in BRAIN_AREAS:
        ot.Mkdir(area_dir(savepath, area), mute=True)


def build_object_space_step1(force=False):
    """NSD1k → AlexNet fc6 → 50D PCA basis."""
    cache = shared_path(savepath, 'step1')
    if os.path.isfile(cache) and not force:
        d = np.load(cache, allow_pickle=True)
        if int(d['n_dim']) != N_DIM:
            raise ValueError(f'cache n_dim={int(d["n_dim"])}, expected {N_DIM}; delete {cache}')
        print(f'[step1] loaded: {cache}')
        return cache

    img_paths = sorted(Path(nsd_figpath).glob('*.bmp')) + sorted(Path(nsd_figpath).glob('*.jpg'))
    img_paths = [str(p) for p in img_paths]
    assert len(img_paths) == 1000, f'expected 1000 NSD images, got {len(img_paths)}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
    buf = []

    def _hook(m, inp, out):
        buf.append(out.detach().cpu())

    h = model.classifier[1].register_forward_hook(_hook)
    fc6 = np.zeros((len(img_paths), 4096), np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), BATCH), desc='AlexNet fc6 (NSD1k)'):
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in img_paths[i:i + BATCH]]
            buf.clear()
            model(torch.stack(imgs).to(device))
            fc6[i:i + BATCH] = buf[0].numpy()
    h.remove()

    pca_full = PCA().fit(fc6)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ev_ratio = pca_full.explained_variance_ratio_[:N_DIM]
    pc_mean = pca_full.mean_.astype(np.float32)
    pc_components = pca_full.components_[:N_DIM].astype(np.float32)
    coords = pca_full.transform(fc6)[:, :N_DIM]

    np.savez(cache, fc6=fc6, cumvar=cumvar, ev_ratio=ev_ratio, n_dim=N_DIM, coords=coords,
             pc_mean=pc_mean, pc_components=pc_components,
             img_paths=np.array(img_paths, dtype=object))
    print(f'[step1] saved: {cache}  ({cumvar[N_DIM - 1]:.1%} variance in {N_DIM} PCs)')
    return cache


def embed_metamers_step2(force=False):
    """Metamer1k fc6 projected onto NSD1k PCA basis."""
    cache1 = shared_path(savepath, 'step1')
    cache2 = shared_path(savepath, 'step2')
    if os.path.isfile(cache2) and not force:
        print(f'[step2] loaded: {cache2}')
        return cache2

    d1 = np.load(cache1, allow_pickle=True)
    pc_mean, pc_components = d1['pc_mean'], d1['pc_components']
    meta_paths = [ot.Join(metamer_figpath, f'{i:04d}.jpg') for i in range(1, N_METAMER + 1)]
    assert all(os.path.isfile(p) for p in meta_paths[:3] + meta_paths[-3:]), 'metamer images not found'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
    buf = []

    def _hook(m, inp, out):
        buf.append(out.detach().cpu())

    h = model.classifier[1].register_forward_hook(_hook)
    meta_fc6 = np.zeros((N_METAMER, 4096), np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, N_METAMER, BATCH), desc='AlexNet fc6 (metamer1k)'):
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in meta_paths[i:i + BATCH]]
            buf.clear()
            model(torch.stack(imgs).to(device))
            meta_fc6[i:i + BATCH] = buf[0].numpy()
    h.remove()

    meta_coords = (meta_fc6 - pc_mean) @ pc_components.T
    np.savez(cache2, fc6=meta_fc6, coords=meta_coords,
             img_paths=np.array(meta_paths, dtype=object))
    print(f'[step2] saved: {cache2}')
    return cache2


def _fit_shuffle_axis(meta_coords, obj_ids, n_dim=N_DIM):
    feats, ys = [], []
    per_obj_raw = {}
    for obj in obj_ids:
        raw = meta_coords[obj, :n_dim]
        per_obj_raw[obj] = raw
        for s in range(5):
            i = obj + s * 40
            feats.append(meta_coords[i, :n_dim] - raw)
            ys.append(float(s))
    F = np.asarray(feats, np.float64)
    y = np.asarray(ys, np.float64)
    w, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
    pred = F @ w
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_group = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return w.astype(np.float32), float(r2_group), per_obj_raw


def _loadings(meta_coords, w, per_obj_raw, obj_ids, n_dim=N_DIM):
    loads = np.zeros(N_METAMER, np.float32)
    for obj in obj_ids:
        for s in range(5):
            i = obj + s * 40
            loads[i] = (meta_coords[i, :n_dim] - per_obj_raw[obj]) @ w
    return loads


def _per_obj_r2(loads, obj_ids):
    r2s = []
    for obj in obj_ids:
        ls = np.array([loads[obj + s * 40] for s in range(5)])
        sh = np.arange(5, dtype=float)
        if ls.var() < 1e-12:
            r2s.append(np.nan)
            continue
        c = np.corrcoef(sh, ls)[0, 1]
        r2s.append(float(c ** 2))
    return np.array(r2s, np.float32)


def fit_shuffle_axes(force=False):
    """Ani / Inani / All shuffle-sensitive axes in 50D object space."""
    cache = shared_path(savepath, 'shuffle_axis')
    if os.path.isfile(cache) and not force:
        print(f'[shuffle_axis] loaded: {cache}')
        return cache

    meta_coords = np.load(shared_path(savepath, 'step2'), allow_pickle=True)['coords']
    ani_objs = list(range(20))
    inani_objs = list(range(20, 40))
    all_objs = list(range(40))

    out = {}
    for tag, objs in (('ani', ani_objs), ('inani', inani_objs), ('all', all_objs)):
        w, r2_g, raw = _fit_shuffle_axis(meta_coords, objs)
        loads = _loadings(meta_coords, w, raw, objs)
        r2_each = _per_obj_r2(loads, objs)
        out[f'w_{tag}'] = w
        out[f'r2_{tag}'] = np.array(r2_g, np.float32)
        out[f'load_{tag}'] = loads
        out[f'r2_{tag}_each'] = r2_each

    np.savez(cache, **out, n_dim=np.array(N_DIM))
    print(f'[shuffle_axis] saved: {cache}')
    for tag in ('ani', 'inani', 'all'):
        print(f'  {tag}: group R²={float(out[f"r2_{tag}"]):.3f}  '
              f'obj median R²={np.nanmedian(out[f"r2_{tag}_each"]):.3f}')
    return cache


def fit_neuron_axes(area, force=False):
    """Per-neuron 50D preferred axis + metamer loadings."""
    cache3 = area_path(savepath, area, 'obj_axis_fit')
    summary_csv = area_path(savepath, area, 'obj_axis_summary')
    if os.path.isfile(cache3) and not force:
        print(f'[{area}] obj_axis_fit loaded: {cache3}')
        return cache3

    d1 = np.load(shared_path(savepath, 'step1'), allow_pickle=True)
    d2 = np.load(shared_path(savepath, 'step2'), allow_pickle=True)
    nsd_coords = d1['coords']
    meta_coords = d2['coords']
    rsp = np.load(rsp_path(cell_rootpath, area))
    assert meta_coords.shape[0] == rsp.shape[1] == N_METAMER

    F_mu = meta_coords.mean(0)
    F_std = meta_coords.std(0)
    F_std[F_std < 1e-8] = 1.0
    F_z = (meta_coords - F_mu) / F_std
    X = np.c_[F_z, np.ones(len(F_z))]

    n_cell = rsp.shape[0]
    axes = np.zeros((n_cell, N_DIM), np.float32)
    bias = np.zeros(n_cell, np.float32)
    r2 = np.zeros(n_cell, np.float32)
    meta_load = np.zeros((n_cell, N_METAMER), np.float32)

    for i in range(n_cell):
        coef, _, _, _ = np.linalg.lstsq(X, rsp[i], rcond=None)
        axes[i] = coef[:N_DIM]
        bias[i] = coef[N_DIM]
        meta_load[i] = F_z @ axes[i]
        pred = meta_load[i] + bias[i]
        v = rsp[i].var()
        r2[i] = 1.0 - (rsp[i] - pred).var() / v if v > 0 else np.nan

    nsd_F_z = (nsd_coords - F_mu) / F_std
    nsd_load_all = nsd_F_z @ axes.T

    hi_cols = [f'nsd_hi_{k}' for k in range(1, N_EXTREME + 1)]
    lo_cols = [f'nsd_lo_{k}' for k in range(1, N_EXTREME + 1)]
    summary_rows = []
    for i in range(n_cell):
        order = np.argsort(nsd_load_all[:, i])
        row = {'cell_idx': i, 'r2': r2[i]}
        for k, j in enumerate(order[-N_EXTREME:][::-1]):
            row[hi_cols[k]] = int(j)
        for k, j in enumerate(order[:N_EXTREME]):
            row[lo_cols[k]] = int(j)
        summary_rows.append(row)

    ot.Mkdir(area_dir(savepath, area), mute=True)
    np.savez(cache3, axes=axes, bias=bias, r2=r2, meta_load=meta_load,
             F_mu=F_mu, F_std=F_std, check_area=np.array(area))
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f'[{area}] {n_cell} cells, median R²={np.nanmedian(r2):.3f}')
    print(f'[{area}] saved: {cache3}')
    print(f'[{area}] saved: {summary_csv}')
    return cache3


def _batch_lin_r2(x, Y):
    x = np.asarray(x, np.float64)
    Y = np.asarray(Y, np.float64)
    X = np.c_[x, np.ones(len(x))]
    coef, _, _, _ = np.linalg.lstsq(X, Y.T, rcond=None)
    pred = X @ coef
    ss_res = ((Y.T - pred) ** 2).sum(0)
    ss_tot = ((Y.T - Y.T.mean(0)) ** 2).sum(0)
    return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan).astype(np.float32)


def fit_shuffle_neuron(area, force=False):
    """Neuron preferred axis vs shuffle axis: angles and R²."""
    cache_n = area_path(savepath, area, 'shuffle_neuron')
    summary_csv = area_path(savepath, area, 'shuffle_neuron_summary')
    if os.path.isfile(cache_n) and not force:
        print(f'[{area}] shuffle_neuron loaded: {cache_n}')
        return cache_n

    d3 = np.load(area_path(savepath, area, 'obj_axis_fit'), allow_pickle=True)
    ds = np.load(shared_path(savepath, 'shuffle_axis'), allow_pickle=True)
    rsp = np.load(rsp_path(cell_rootpath, area))
    n_cell = rsp.shape[0]

    axes_cell = d3['axes']
    F_mu, F_std = d3['F_mu'], d3['F_std']
    r2_obj_axis = d3['r2']
    w_ani, w_inani, w_all = ds['w_ani'], ds['w_inani'], ds['w_all']
    load_ani, load_inani, load_all = ds['load_ani'], ds['load_inani'], ds['load_all']

    idx = np.arange(N_METAMER)
    is_ani = (idx % 40) < 20
    mask_ani, mask_inani = is_ani, ~is_ani

    w_ani_z = w_ani / F_std
    w_inani_z = w_inani / F_std
    w_all_z = w_all / F_std
    u_shuf_ani = w_ani_z / (np.linalg.norm(w_ani_z) + 1e-8)
    u_shuf_inani = w_inani_z / (np.linalg.norm(w_inani_z) + 1e-8)
    u_shuf_all = w_all_z / (np.linalg.norm(w_all_z) + 1e-8)
    u_cell = axes_cell / (np.linalg.norm(axes_cell, axis=1, keepdims=True) + 1e-8)

    cos_ani = np.clip(u_cell @ u_shuf_ani, -1.0, 1.0).astype(np.float32)
    cos_inani = np.clip(u_cell @ u_shuf_inani, -1.0, 1.0).astype(np.float32)
    cos_all = np.clip(u_cell @ u_shuf_all, -1.0, 1.0).astype(np.float32)
    angle_ani = np.degrees(np.arccos(cos_ani)).astype(np.float32)
    angle_inani = np.degrees(np.arccos(cos_inani)).astype(np.float32)
    angle_all = np.degrees(np.arccos(cos_all)).astype(np.float32)

    r2_shuf_ani = _batch_lin_r2(load_ani[mask_ani], rsp[:, mask_ani])
    r2_shuf_inani = _batch_lin_r2(load_inani[mask_inani], rsp[:, mask_inani])
    r2_shuf_all = _batch_lin_r2(load_all, rsp)

    summary = pd.DataFrame({
        'cell_idx': np.arange(n_cell),
        'angle_ani': angle_ani,
        'angle_inani': angle_inani,
        'angle_all': angle_all,
        'cos_ani': cos_ani,
        'cos_inani': cos_inani,
        'cos_all': cos_all,
        'r2_shuf_ani': r2_shuf_ani,
        'r2_shuf_inani': r2_shuf_inani,
        'r2_shuf_all': r2_shuf_all,
        'r2_obj_axis': r2_obj_axis,
    })
    np.savez(cache_n, angle_ani=angle_ani, angle_inani=angle_inani, angle_all=angle_all,
             cos_ani=cos_ani, cos_inani=cos_inani, cos_all=cos_all,
             r2_shuf_ani=r2_shuf_ani, r2_shuf_inani=r2_shuf_inani, r2_shuf_all=r2_shuf_all,
             check_area=np.array(area))
    summary.to_csv(summary_csv, index=False)
    print(f'[{area}] shuffle_neuron median R² ani/inani/all = '
          f'{np.nanmedian(r2_shuf_ani):.3f} / {np.nanmedian(r2_shuf_inani):.3f} / '
          f'{np.nanmedian(r2_shuf_all):.3f}')
    print(f'[{area}] saved: {cache_n}')
    return cache_n


def _stimulus_groups():
    idx = np.arange(N_METAMER)
    within = idx % 200
    shuffle = within // 40
    is_ani = (within % 40) < 20
    parent = within % 40
    group_idx = np.full((N_OBJ, N_SHUF, N_METAMER // (N_OBJ * N_SHUF)), -1, dtype=int)
    for o in range(N_OBJ):
        for s in range(N_SHUF):
            hits = np.where((parent == o) & (shuffle == s))[0]
            group_idx[o, s, :len(hits)] = hits
    return shuffle, is_ani, parent, group_idx


def fit_mediation(area, force=False):
    """Shuffle slope mediation: load vs response, incremental ΔR²."""
    cache_m = area_path(savepath, area, 'mediation')
    if os.path.isfile(cache_m) and not force:
        print(f'[{area}] mediation loaded: {cache_m}')
        return cache_m

    d3 = np.load(area_path(savepath, area, 'obj_axis_fit'), allow_pickle=True)
    rsp = np.load(rsp_path(cell_rootpath, area))
    meta_load = d3['meta_load'].astype(np.float32)
    r2_load = d3['r2'].astype(np.float32)
    n_cell = rsp.shape[0]

    shuffle, is_ani, parent, group_idx = _stimulus_groups()
    gi = group_idx.reshape(N_OBJ * N_SHUF, -1)
    avg_load = meta_load[:, gi].mean(-1).reshape(n_cell, N_OBJ, N_SHUF)
    avg_rsp = rsp[:, gi].mean(-1).reshape(n_cell, N_OBJ, N_SHUF)

    shuf_c = np.arange(N_SHUF, dtype=np.float32) - 2.0
    slope_load = (avg_load * shuf_c).sum(-1) / 10.0
    slope_rsp = (avg_rsp * shuf_c).sum(-1) / 10.0

    sl_mu = slope_load.mean(1, keepdims=True)
    sr_mu = slope_rsp.mean(1, keepdims=True)
    sl_c = slope_load - sl_mu
    sr_c = slope_rsp - sr_mu
    cov = (sl_c * sr_c).sum(1)
    std_l = np.sqrt((sl_c ** 2).sum(1))
    std_r = np.sqrt((sr_c ** 2).sum(1))
    denom = std_l * std_r
    pearson_r = np.where(denom > 1e-8, cov / denom, np.nan).astype(np.float32)

    al_c = avg_load - avg_load.mean(-1, keepdims=True)
    ar_c = avg_rsp - avg_rsp.mean(-1, keepdims=True)
    cor_per_obj = np.clip(
        (al_c * ar_c).sum(-1) / (
            np.sqrt((al_c ** 2).sum(-1)) * np.sqrt((ar_c ** 2).sum(-1)) + 1e-16
        ), -1, 1
    ).astype(np.float32)

    shuf_f = shuffle.astype(np.float32)
    shuf_z = (shuf_f - shuf_f.mean()) / (shuf_f.std() + 1e-8)
    r2_full = np.full(n_cell, np.nan, np.float32)
    for i in range(n_cell):
        X = np.c_[meta_load[i], shuf_z, np.ones(N_METAMER)]
        coef, _, _, _ = np.linalg.lstsq(X, rsp[i], rcond=None)
        pred = X @ coef
        ss_res = ((rsp[i] - pred) ** 2).sum()
        ss_tot = rsp[i].var() * N_METAMER
        r2_full[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    delta_r2 = r2_full - r2_load

    np.savez(
        cache_m,
        avg_load=avg_load,
        avg_rsp=avg_rsp,
        slope_load=slope_load,
        slope_rsp=slope_rsp,
        pearson_r=pearson_r,
        cor_per_obj=cor_per_obj,
        r2_full=r2_full,
        delta_r2=delta_r2,
        r2_load=r2_load,
        check_area=np.array(area),
    )
    print(f'[{area}] mediation median Pearson r={np.nanmedian(pearson_r):.3f}  '
          f'median ΔR²={np.nanmedian(delta_r2):.4f}')
    print(f'[{area}] saved: {cache_m}')
    return cache_m


def run_all(force=False):
    _ensure_dirs()
    readme_src = Path(__file__).with_name('README.md')
    if readme_src.is_file():
        shutil.copy2(readme_src, ot.Join(savepath, 'README.md'))
    build_object_space_step1(force=force)
    embed_metamers_step2(force=force)
    fit_shuffle_axes(force=force)
    for area in BRAIN_AREAS:
        rsp_p = rsp_path(cell_rootpath, area)
        if not os.path.isfile(rsp_p):
            print(f'[{area}] SKIP — missing {rsp_p}')
            continue
        fit_neuron_axes(area, force=force)
        fit_shuffle_neuron(area, force=force)
        fit_mediation(area, force=force)
    from obj_space_figures import generate_all_figures
    generate_all_figures(savepath, cell_rootpath, show=False)
    print('Done.')

#%%
if __name__ == '__main__':
    run_all(force=False)





 