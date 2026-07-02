r'''
使用 Metamer_NSD_2k 数据集中的 NSD 部分训练 AlexNet fc6 encoder。

核心思想：
1. 只使用 Metamer_NSD_2k 中的 NSD 响应。
   该数据集中前 1000 列是 metamer，最后 1000 列是 NSD。
   本脚本只取最后 1000 个 NSD condition 进行拟合和交叉验证。

2. 图片表征使用 NSD1000 的 AlexNet fc6。
   先对 1000 张 NSD 图片提取 4096D fc6，再在这 1000 张图片上做一次 PCA。
   PC 数目在 1-20 之间 grid search。
   本脚本不读取 object-space 相关缓存；NSD fc6 会从图片重新计算。

3. 响应和图片表征按 NSD object_id 对齐。
   如果 stim_layout.npz 中 NSD 顺序不是 1..1000，会自动重排响应/特征。

4. 交叉验证使用 20-fold KFold。
   每个 NSD 图片是一个独立 condition，因此直接在 1000 个图片 id 上分 fold。

5. 结果指标：
   fit_r2_adj = fit_r2 / ceiling_index
   cv_r2_adj = cv_r2 / ceiling_index
   PC grid search 的曲线和最佳 PC 选择都使用 adjusted R2。

保存目录：
E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Encoder_fc6

数据目录：
E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_NSD_2k
'''

#%%
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


#%% 参数
SAVEPATH = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Encoder_fc6')
DATAPATH = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_NSD_2k')
NSD_STIMPATH = Path(r'E:\#Stimsets\NSD1000')

NSD_PCA_CACHE = SAVEPATH / 'alexnet_nsd1k_global_fc6_pca.npz'

N_METAMER = 1000
N_NSD = 1000
N_PC = 20
PC_GRID = np.arange(1, N_PC + 1)
N_CV_FOLDS = 20
CV_SEED = 42
BATCH_SIZE = 32

WINDOW_S = 0.17
CONVERT_TO_HZ = False
FORCE_FEATURE_REBUILD = True

SUMMARY_CSV = SAVEPATH / 'encoding_model_nsd_summary.csv'
GRID_SEARCH_CSV = SAVEPATH / 'encoding_model_nsd_pc_grid_search.csv'

SAVEPATH.mkdir(parents=True, exist_ok=True)


#%% NSD 图片和 PCA
def nsd_object_id_from_path(path):
    """Map NSD1000 filename to 1-based object id, e.g. 50001.jpg -> 1."""
    stem = Path(path).stem
    if not stem.isdigit():
        return None
    value = int(stem)
    return value - 50000 if value >= 50000 else value


def nsd_image_paths():
    paths = sorted(NSD_STIMPATH.glob('*.bmp')) + sorted(NSD_STIMPATH.glob('*.jpg'))
    if len(paths) != N_NSD:
        raise FileNotFoundError(f'Expected {N_NSD} NSD images in {NSD_STIMPATH}, got {len(paths)}.')
    return paths


def extract_fc6_features(img_paths):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    import torch
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()

    def _extract(batch_tensor):
        buf = []

        def _hook(_module, _inp, out):
            # classifier[2] is inplace ReLU, so clone fc6 immediately.
            buf.append(out.detach().cpu().clone())

        handle = model.classifier[1].register_forward_hook(_hook)
        model(batch_tensor)
        handle.remove()
        return buf[0]

    sample = _extract(preprocess(Image.open(img_paths[0]).convert('RGB')).unsqueeze(0).to(device))
    fc6 = np.zeros((len(img_paths), int(sample.shape[1])), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, len(img_paths), BATCH_SIZE), desc='AlexNet fc6 NSD'):
            batch_paths = img_paths[start:start + BATCH_SIZE]
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
            fc6[start:start + len(imgs)] = _extract(torch.stack(imgs).to(device)).numpy()
    return fc6


def build_nsd_pca_from_fc6(fc6, img_paths):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=N_PC).fit(fc6)
    coords = pca.transform(fc6).astype(np.float32)
    np.savez(
        NSD_PCA_CACHE,
        fc6=fc6,
        coords=coords,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        n_pc=np.array(N_PC),
        img_paths=np.array([str(p) for p in img_paths], dtype=object),
    )
    return np.load(NSD_PCA_CACHE, allow_pickle=True)


def load_or_build_nsd_pca(force=False):
    if NSD_PCA_CACHE.is_file() and not force:
        d = np.load(NSD_PCA_CACHE, allow_pickle=True)
        if int(d['n_pc']) == N_PC:
            print(f'Loaded NSD fc6 PCA cache: {NSD_PCA_CACHE}')
            return d['coords'].astype(np.float32), list(d['img_paths']), d
        if 'fc6' in d.files:
            warnings.warn(f'Rebuilding NSD PCA with n_pc={N_PC} from cached fc6.')
            d = build_nsd_pca_from_fc6(d['fc6'], d['img_paths'])
            return d['coords'].astype(np.float32), list(d['img_paths']), d

    img_paths = nsd_image_paths()
    if force:
        print('Force rebuilding NSD fc6 PCA from image files; no object-space cache is used.')
    fc6 = extract_fc6_features(img_paths)
    d = build_nsd_pca_from_fc6(fc6, img_paths)
    print(f'Saved NSD fc6 PCA cache: {NSD_PCA_CACHE}')
    return d['coords'].astype(np.float32), list(d['img_paths']), d


def align_nsd_coords_and_rsp(nsd_coords, img_paths, rsp_nsd, nsd_object_ids):
    """
    Align NSD PCA coords and responses by 1-based object id.
    Returns coords and responses in the order given by stim_layout's NSD object_id.
    """
    if rsp_nsd.shape[1] != len(nsd_object_ids):
        raise ValueError(
            f'NSD response columns ({rsp_nsd.shape[1]}) do not match '
            f'layout object ids ({len(nsd_object_ids)}).'
        )

    ids_from_paths = np.array([nsd_object_id_from_path(p) for p in img_paths], dtype=np.int32)
    if np.any(ids_from_paths <= 0):
        raise ValueError('Could not parse NSD object ids from image paths.')

    coords_by_oid = np.full((N_NSD, nsd_coords.shape[1]), np.nan, dtype=np.float32)
    for idx, object_id in enumerate(ids_from_paths):
        if 1 <= object_id <= N_NSD:
            coords_by_oid[object_id - 1] = nsd_coords[idx]

    export_oids = np.asarray(nsd_object_ids, dtype=np.int32)
    coords_aligned = coords_by_oid[export_oids - 1]
    rsp_aligned = rsp_nsd

    if np.any(~np.isfinite(coords_aligned)):
        missing = np.where(~np.isfinite(coords_aligned).any(axis=1))[0] + 1
        raise ValueError(f'Missing NSD coords for object ids: {missing[:10]}...')

    return coords_aligned.astype(np.float32), rsp_aligned.astype(np.float64), export_oids


#%% 数据读取
def available_areas():
    out = []
    for p in sorted(DATAPATH.iterdir()):
        if not p.is_dir():
            continue
        if (p / 'avr_rsp.npy').is_file() and (p / 'cell_site_info.csv').is_file():
            out.append(p.name)
    if not out:
        raise FileNotFoundError(f'No area folders with avr_rsp.npy found in {DATAPATH}.')
    return out


def nsd_slice_and_ids():
    layout_path = DATAPATH / 'stim_layout.npz'
    if not layout_path.is_file():
        warnings.warn('stim_layout.npz not found; using last 1000 columns and object_id=1..1000.')
        return slice(N_METAMER, N_METAMER + N_NSD), np.arange(1, N_NSD + 1, dtype=np.int32)

    layout = np.load(layout_path, allow_pickle=True)
    if 'slice_nsd' in layout.files:
        values = np.asarray(layout['slice_nsd'], dtype=int).ravel()
        nsd_slice = slice(int(values[0]), int(values[1]))
    else:
        nsd_slice = slice(N_METAMER, N_METAMER + N_NSD)

    if 'object_id' in layout.files:
        object_ids = np.asarray(layout['object_id'][nsd_slice], dtype=np.int32)
    else:
        object_ids = np.arange(1, N_NSD + 1, dtype=np.int32)

    if len(object_ids) != N_NSD:
        raise ValueError(f'Expected {N_NSD} NSD object ids, got {len(object_ids)}.')
    return nsd_slice, object_ids


def load_area_nsd_data(area, nsd_slice):
    area_path = DATAPATH / area
    rsp_all = np.load(area_path / 'avr_rsp.npy').astype(np.float64)
    cell_info = pd.read_csv(area_path / 'cell_site_info.csv')

    if len(cell_info) != rsp_all.shape[0]:
        raise ValueError(
            f'{area}: cell_site_info rows ({len(cell_info)}) do not match '
            f'avr_rsp cells ({rsp_all.shape[0]}).'
        )
    if 'ceiling_index' not in cell_info.columns:
        raise KeyError(f'{area}: cell_site_info.csv lacks column "ceiling_index".')

    if rsp_all.shape[1] < nsd_slice.stop:
        raise ValueError(f'{area}: expected at least {nsd_slice.stop} response columns, got {rsp_all.shape[1]}.')

    rsp_nsd = rsp_all[:, nsd_slice].astype(np.float64)
    if CONVERT_TO_HZ:
        rsp_nsd = rsp_nsd / WINDOW_S

    ceiling = pd.to_numeric(cell_info['ceiling_index'], errors='coerce').to_numpy(np.float64)
    return rsp_nsd, cell_info, ceiling


#%% 模型与 20-fold CV
def standardize_train_test(x_train, x_test=None):
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x_train_z = (x_train - mu) / sd
    if x_test is None:
        return x_train_z, mu, sd
    return x_train_z, (x_test - mu) / sd, mu, sd


def fit_linear_multioutput(x, y):
    x_aug = np.c_[x, np.ones(len(x))]
    coef_aug, _, _, _ = np.linalg.lstsq(x_aug, y.T, rcond=None)
    pred = x_aug @ coef_aug
    return coef_aug[:-1].astype(np.float32), coef_aug[-1].astype(np.float32), pred


def r2_per_cell(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan).astype(np.float32)


def adjust_r2_by_ceiling(r2, ceiling):
    ceiling = np.asarray(ceiling, dtype=np.float64)
    out = np.full_like(r2, np.nan, dtype=np.float32)
    valid = np.isfinite(r2) & np.isfinite(ceiling) & (ceiling > 0)
    out[valid] = (r2[valid] / ceiling[valid]).astype(np.float32)
    return out


def image_kfold_ids(n_sample):
    from sklearn.model_selection import KFold

    fold_ids = np.full(n_sample, -1, dtype=int)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold, (_, test_idx) in enumerate(kf.split(np.arange(n_sample))):
        fold_ids[test_idx] = fold
    return fold_ids


def fit_group_model_kfold(x, y, ceiling):
    x_z, x_mean, x_std = standardize_train_test(x)
    weights, bias, y_fit = fit_linear_multioutput(x_z, y)
    fit_r2 = r2_per_cell(y.T, y_fit)

    fold_ids = image_kfold_ids(x.shape[0])
    y_cv = np.full_like(y.T, np.nan, dtype=np.float64)
    for fold in range(N_CV_FOLDS):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        x_tr, x_te, _, _ = standardize_train_test(x[train_mask], x[test_mask])
        x_tr_aug = np.c_[x_tr, np.ones(len(x_tr))]
        coef_aug, _, _, _ = np.linalg.lstsq(x_tr_aug, y[:, train_mask].T, rcond=None)
        y_cv[test_mask] = np.c_[x_te, np.ones(len(x_te))] @ coef_aug

    cv_r2 = r2_per_cell(y.T, y_cv)
    return {
        'weights': weights,
        'bias': bias,
        'x_mean': x_mean.astype(np.float32),
        'x_std': x_std.astype(np.float32),
        'y_fit': y_fit.T.astype(np.float32),
        'y_cv': y_cv.T.astype(np.float32),
        'fold_ids': fold_ids.astype(np.int32),
        'fit_r2': fit_r2,
        'cv_r2': cv_r2,
        'fit_r2_adj': adjust_r2_by_ceiling(fit_r2, ceiling),
        'cv_r2_adj': adjust_r2_by_ceiling(cv_r2, ceiling),
    }


def save_result(area, object_ids, result, cell_info, ceiling, tag='best'):
    area_dir = SAVEPATH / area
    area_dir.mkdir(parents=True, exist_ok=True)
    n_model_pc = int(result['weights'].shape[0])
    stem = f'{area}_nsd_20fold_{tag}_fc6pc{n_model_pc}'
    npz_path = area_dir / f'{stem}.npz'
    csv_path = area_dir / f'{stem}_cell_metrics.csv'

    np.savez(
        npz_path,
        weights=result['weights'],
        bias=result['bias'],
        x_mean=result['x_mean'],
        x_std=result['x_std'],
        y_fit=result['y_fit'],
        y_cv=result['y_cv'],
        fold_ids=result['fold_ids'],
        fit_r2=result['fit_r2'],
        cv_r2=result['cv_r2'],
        fit_r2_adj=result['fit_r2_adj'],
        cv_r2_adj=result['cv_r2_adj'],
        ceiling_index=ceiling.astype(np.float32),
        object_id=object_ids.astype(np.int32),
        feature_n_pc=np.array(N_PC),
        model_n_pc=np.array(n_model_pc),
        n_cv_folds=np.array(N_CV_FOLDS),
        cv_seed=np.array(CV_SEED),
    )

    metrics = pd.DataFrame({
        'cell_idx': np.arange(len(ceiling)),
        'ceiling_index': ceiling,
        'fit_r2': result['fit_r2'],
        'fit_r2_adj': result['fit_r2_adj'],
        'cv_r2': result['cv_r2'],
        'cv_r2_adj': result['cv_r2_adj'],
    })
    passthrough_cols = [
        col for col in ('cell_id', 'site', 'area', 'dprime_face', 'dprime_body')
        if col in cell_info.columns
    ]
    if passthrough_cols:
        metrics = pd.concat([cell_info[passthrough_cols].reset_index(drop=True), metrics], axis=1)
    metrics.to_csv(csv_path, index=False)
    return npz_path, csv_path


def summarize_result(area, n_model_pc, result, n_cell):
    return {
        'area': area,
        'dataset': 'nsd',
        'n_condition': N_NSD,
        'n_cv_folds': N_CV_FOLDS,
        'cv_seed': CV_SEED,
        'feature_n_pc': N_PC,
        'model_n_pc': n_model_pc,
        'n_cell': n_cell,
        'median_fit_r2': np.nanmedian(result['fit_r2']),
        'median_fit_r2_adj': np.nanmedian(result['fit_r2_adj']),
        'median_cv_r2': np.nanmedian(result['cv_r2']),
        'median_cv_r2_adj': np.nanmedian(result['cv_r2_adj']),
        'mean_fit_r2': np.nanmean(result['fit_r2']),
        'mean_fit_r2_adj': np.nanmean(result['fit_r2_adj']),
        'mean_cv_r2': np.nanmean(result['cv_r2']),
        'mean_cv_r2_adj': np.nanmean(result['cv_r2_adj']),
    }


def best_score(result):
    score = float(np.nanmedian(result['cv_r2_adj']))
    return score if np.isfinite(score) else -np.inf


def explained_variance_text(pca_cache):
    if 'explained_variance_ratio' in pca_cache.files:
        value = np.sum(pca_cache['explained_variance_ratio'][:N_PC])
        return f'{value:.2%}'
    if 'ev_ratio' in pca_cache.files:
        value = np.sum(pca_cache['ev_ratio'][:N_PC])
        return f'{value:.2%}'
    return 'unknown'


def plot_area_grid_search(area, rows):
    import matplotlib.pyplot as plt

    area_dir = SAVEPATH / area
    area_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values('model_n_pc')
    best_row = df.loc[df['median_cv_r2_adj'].idxmax()]

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    ax.plot(df['model_n_pc'], df['median_fit_r2_adj'], '-o', ms=3, label='fit R2_adj')
    ax.plot(df['model_n_pc'], df['median_cv_r2_adj'], '-o', ms=3, label='20-fold CV R2_adj')
    ax.axvline(best_row['model_n_pc'], color='0.4', ls='--', lw=0.8)
    ax.set_title(f'{area} NSD encoder: best PC={int(best_row["model_n_pc"])}')
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Median R2_adj across cells')
    ax.set_xticks(PC_GRID)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = area_dir / f'{area}_nsd_20fold_pc_grid_search_r2_adj.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


#%% 主流程
def run_all():
    nsd_coords_raw, img_paths, pca_cache = load_or_build_nsd_pca(force=FORCE_FEATURE_REBUILD)
    nsd_slice, nsd_object_ids = nsd_slice_and_ids()
    areas = available_areas()

    print(
        f'NSD feature cache ready: coords={nsd_coords_raw.shape}, '
        f'first {N_PC} PCs explain {explained_variance_text(pca_cache)} fc6 variance.'
    )
    print(f'Areas found in Metamer_NSD_2k: {areas}')

    best_rows = []
    grid_rows = []

    for area in tqdm(areas, desc='area'):
        try:
            rsp_nsd_raw, cell_info, ceiling = load_area_nsd_data(area, nsd_slice)
        except FileNotFoundError as exc:
            warnings.warn(f'Skip {area}: {exc}')
            continue

        nsd_coords, rsp_nsd, object_ids = align_nsd_coords_and_rsp(
            nsd_coords_raw, img_paths, rsp_nsd_raw, nsd_object_ids
        )

        area_rows = []
        best_result = None
        best_row = None
        best_metric = -np.inf

        for n_model_pc in PC_GRID:
            x = nsd_coords[:, :n_model_pc]
            result = fit_group_model_kfold(x, rsp_nsd, ceiling)
            row = summarize_result(area, n_model_pc, result, n_cell=rsp_nsd.shape[0])
            area_rows.append(row)
            grid_rows.append(row.copy())

            score = best_score(result)
            if score > best_metric:
                best_metric = score
                best_result = result
                best_row = row.copy()

        npz_path, csv_path = save_result(area, object_ids, best_result, cell_info, ceiling, tag='best')
        best_row['npz_path'] = str(npz_path)
        best_row['cell_metrics_csv'] = str(csv_path)
        best_rows.append(best_row)

        fig_path = plot_area_grid_search(area, area_rows)
        print(
            f'[{area} NSD] best PC={int(best_row["model_n_pc"])}, '
            f'median fit R2_adj={best_row["median_fit_r2_adj"]:.3f}, '
            f'20-fold CV R2_adj={best_row["median_cv_r2_adj"]:.3f}'
        )
        print(f'[{area}] saved PC grid curve: {fig_path}')

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(GRID_SEARCH_CSV, index=False)

    summary = pd.DataFrame(best_rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f'Saved grid search: {GRID_SEARCH_CSV}')
    print(f'Saved summary: {SUMMARY_CSV}')
    return summary


if __name__ == '__main__':
    run_all()


#%%

