r'''
使用 AlexNet fc6 建立 encoding model，并使用 object-level 5-fold 交叉验证。

核心思想：
1. 先对全部 1000 张 Metamer1k 图片提取 AlexNet fc6 表征。
   得到 1000 × 4096 的 fc6 矩阵。

2. PCA 只在全部 1000 张图片的 fc6 矩阵上做一次。
   后续所有脑区、ani/inani/all 模型都共用这个 PCA 空间。

3. avr_rsp.npy 的列顺序按：
   5 repeat × (5 shuffle level × 40 image)
   本脚本先平均同一 condition 的 5 个 repeat，得到 200 个 condition-level 响应。

4. 合并全部 5 个 shuffle level：
   ani: 20 个 object × 5 shuffle level = 100 个 condition
   inani: 20 个 object × 5 shuffle level = 100 个 condition
   all: 40 个 object × 5 shuffle level = 200 个 condition

5. 交叉验证使用 object-level 5-fold：
   同一个 object 的 5 个 shuffle-level condition 必须在同一个 fold。
   这样测试某个 object 时，训练集中不会出现这个 object 的其它 shuffle 版本。

6. 对 PC 数目 1-20 做 grid search。
   对每个 area / ani-inani-all 组合，保存完整 grid search 表、fit/CV R2 曲线图，
   并保存 median cv_r2 最好的模型。

保存目录：
E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Encoder_fc6

数据目录：
E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k
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
DATAPATH = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k')
STIMPATH = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Stimsets\Metamer1k')

BRAIN_AREAS = ('ML', 'MSB', 'AL', 'ASB')
SHUFFLE_LABELS = ('Raw', 'S1', 'S2', 'S3', 'S4')
SCOPES = {
    'ani': np.arange(0, 20),
    'inani': np.arange(20, 40),
    'all': np.arange(0, 40),
}

N_REPEAT = 5
N_SHUF = 5
N_OBJ = 40
N_COND = N_SHUF * N_OBJ
N_METAMER = N_REPEAT * N_COND
N_PC = 20
PC_GRID = np.arange(1, N_PC + 1)
N_CV_FOLDS = 5
CV_SEED = 42
BATCH_SIZE = 32

WINDOW_S = 0.17
CONVERT_TO_HZ = False
FORCE_FEATURE_REBUILD = False

GLOBAL_PCA_CACHE = SAVEPATH / 'alexnet_metamer1k_global_fc6_pca.npz'
SUMMARY_CSV = SAVEPATH / 'encoding_model_5fold_fc6_summary.csv'
GRID_SEARCH_CSV = SAVEPATH / 'encoding_model_5fold_fc6_pc_grid_search.csv'

SAVEPATH.mkdir(parents=True, exist_ok=True)


#%% 刺激索引
def condition_table():
    cond = np.arange(N_COND)
    return pd.DataFrame({
        'cond_idx': cond,
        'shuffle': cond // N_OBJ,
        'shuffle_label': [SHUFFLE_LABELS[i] for i in cond // N_OBJ],
        'object_id': cond % N_OBJ,
        'scope': np.where((cond % N_OBJ) < 20, 'ani', 'inani'),
    })


COND_TABLE = condition_table()


def condition_indices(scope):
    if scope not in SCOPES:
        raise ValueError(f'Unknown scope={scope!r}. Expected one of {list(SCOPES)}.')
    object_ids = SCOPES[scope]
    mask = np.isin(COND_TABLE['object_id'].to_numpy(), object_ids)
    idx = COND_TABLE.loc[mask, 'cond_idx'].to_numpy(dtype=int)
    expected = len(object_ids) * N_SHUF
    if len(idx) != expected:
        raise ValueError(f'scope={scope}: expected {expected} conditions, got {len(idx)}.')
    return idx


#%% 全局 AlexNet fc6 PCA
def image_paths_metamer1k():
    paths = [STIMPATH / f'{i:04d}.jpg' for i in range(1, N_METAMER + 1)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing metamer image files, first missing file: {missing[0]}')
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
            buf.append(out.detach().cpu().clone())

        handle = model.classifier[1].register_forward_hook(_hook)
        model(batch_tensor)
        handle.remove()
        return buf[0]

    sample = _extract(preprocess(Image.open(img_paths[0]).convert('RGB')).unsqueeze(0).to(device))
    fc6 = np.zeros((len(img_paths), int(sample.shape[1])), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, len(img_paths), BATCH_SIZE), desc='AlexNet fc6'):
            batch_paths = img_paths[start:start + BATCH_SIZE]
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
            fc6[start:start + len(imgs)] = _extract(torch.stack(imgs).to(device)).numpy()
    return fc6


def build_pca_from_fc6(fc6, img_paths):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=N_PC).fit(fc6)
    coords = pca.transform(fc6).astype(np.float32)
    cond_coords = coords.reshape(N_REPEAT, N_COND, N_PC).mean(axis=0).astype(np.float32)
    np.savez(
        GLOBAL_PCA_CACHE,
        fc6=fc6,
        coords=coords,
        cond_coords=cond_coords,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        n_pc=np.array(N_PC),
        img_paths=np.array([str(p) for p in img_paths], dtype=object),
    )
    return np.load(GLOBAL_PCA_CACHE, allow_pickle=True)


def load_or_build_global_fc6_pca(force=False):
    if GLOBAL_PCA_CACHE.is_file() and not force:
        d = np.load(GLOBAL_PCA_CACHE, allow_pickle=True)
        if int(d['n_pc']) != N_PC:
            if 'fc6' not in d.files:
                raise ValueError(
                    f'{GLOBAL_PCA_CACHE} has n_pc={int(d["n_pc"])}, '
                    f'but current N_PC={N_PC}, and no cached fc6 is available.'
                )
            warnings.warn(f'Rebuilding PCA with n_pc={N_PC} from cached fc6.')
            d = build_pca_from_fc6(d['fc6'], d['img_paths'] if 'img_paths' in d.files else [])
        coords = d['coords']
        cond_coords = (
            d['cond_coords'] if 'cond_coords' in d.files
            else coords.reshape(N_REPEAT, N_COND, N_PC).mean(axis=0).astype(np.float32)
        )
        print(f'Loaded global fc6 PCA cache: {GLOBAL_PCA_CACHE}')
        return coords, cond_coords, d

    img_paths = image_paths_metamer1k()
    fc6 = extract_fc6_features(img_paths)
    d = build_pca_from_fc6(fc6, img_paths)
    print(f'Saved global fc6 PCA cache: {GLOBAL_PCA_CACHE}')
    return d['coords'], d['cond_coords'], d


#%% 数据读取
def load_area_data(area):
    area_path = DATAPATH / area
    rsp = np.load(area_path / 'avr_rsp.npy').astype(np.float64)
    cell_info = pd.read_csv(area_path / 'cell_site_info.csv')

    if len(cell_info) != rsp.shape[0]:
        raise ValueError(
            f'{area}: cell_site_info rows ({len(cell_info)}) do not match '
            f'avr_rsp cells ({rsp.shape[0]}).'
        )
    if 'ceiling_index' not in cell_info.columns:
        raise KeyError(f'{area}: cell_site_info.csv lacks column "ceiling_index".')

    if CONVERT_TO_HZ:
        rsp = rsp / WINDOW_S

    if rsp.shape[1] == N_METAMER:
        cond_rsp = rsp.reshape(rsp.shape[0], N_REPEAT, N_COND).mean(axis=1)
    elif rsp.shape[1] == N_COND:
        cond_rsp = rsp
    else:
        raise ValueError(
            f'{area}: expected avr_rsp columns {N_METAMER} or {N_COND}, got {rsp.shape[1]}.'
        )

    ceiling = pd.to_numeric(cell_info['ceiling_index'], errors='coerce').to_numpy(np.float64)
    return cond_rsp, cell_info, ceiling


#%% 模型与 5-fold CV
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


def object_5fold_ids(group_ids):
    from sklearn.model_selection import KFold

    unique_groups = np.array(sorted(np.unique(group_ids)))
    if len(unique_groups) < N_CV_FOLDS:
        raise ValueError(f'Need at least {N_CV_FOLDS} objects, got {len(unique_groups)}.')

    fold_ids = np.full(len(group_ids), -1, dtype=int)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=CV_SEED)
    for fold, (_, test_group_pos) in enumerate(kf.split(unique_groups)):
        test_groups = unique_groups[test_group_pos]
        fold_ids[np.isin(group_ids, test_groups)] = fold
    if np.any(fold_ids < 0):
        raise RuntimeError('Some conditions were not assigned to a CV fold.')
    return fold_ids


def fit_group_model_5fold(x_cond, y_cond, group_ids, ceiling):
    x_z, x_mean, x_std = standardize_train_test(x_cond)
    weights, bias, y_fit = fit_linear_multioutput(x_z, y_cond)
    fit_r2 = r2_per_cell(y_cond.T, y_fit)

    fold_ids = object_5fold_ids(group_ids)
    y_cv = np.full_like(y_cond.T, np.nan, dtype=np.float64)
    for fold in range(N_CV_FOLDS):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        x_tr, x_te, _, _ = standardize_train_test(x_cond[train_mask], x_cond[test_mask])
        x_tr_aug = np.c_[x_tr, np.ones(len(x_tr))]
        coef_aug, _, _, _ = np.linalg.lstsq(x_tr_aug, y_cond[:, train_mask].T, rcond=None)
        y_cv[test_mask] = np.c_[x_te, np.ones(len(x_te))] @ coef_aug

    cv_r2 = r2_per_cell(y_cond.T, y_cv)
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


def save_group_result(area, scope, cond_idx, result, cell_info, ceiling, tag='best'):
    area_dir = SAVEPATH / area
    area_dir.mkdir(parents=True, exist_ok=True)
    n_model_pc = int(result['weights'].shape[0])
    stem = f'{area}_allshuffle_{scope}_5fold_{tag}_fc6pc{n_model_pc}'
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
        cond_idx=cond_idx.astype(np.int32),
        object_id=COND_TABLE.loc[cond_idx, 'object_id'].to_numpy(dtype=np.int32),
        shuffle=COND_TABLE.loc[cond_idx, 'shuffle'].to_numpy(dtype=np.int32),
        shuffle_label=COND_TABLE.loc[cond_idx, 'shuffle_label'].to_numpy(dtype=object),
        scope=np.array(scope),
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


def summarize_result(area, scope, cond_idx, group_ids, n_model_pc, result):
    return {
        'area': area,
        'scope': scope,
        'n_condition': len(cond_idx),
        'n_cv_groups': len(np.unique(group_ids)),
        'cv_group': 'object_id',
        'n_cv_folds': N_CV_FOLDS,
        'cv_seed': CV_SEED,
        'n_shuffle': N_SHUF,
        'n_repeat': N_REPEAT,
        'feature_n_pc': N_PC,
        'model_n_pc': n_model_pc,
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


def plot_area_grid_search(area, area_grid_rows):
    import matplotlib.pyplot as plt

    area_dir = SAVEPATH / area
    area_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(area_grid_rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=150, sharey=False)

    for ax, scope in zip(axes, SCOPES):
        sub = df[df['scope'] == scope].sort_values('model_n_pc')
        ax.plot(sub['model_n_pc'], sub['median_fit_r2_adj'], '-o', ms=3, label='fit R2_adj')
        ax.plot(sub['model_n_pc'], sub['median_cv_r2_adj'], '-o', ms=3, label='5-fold CV R2_adj')
        best_row = sub.loc[sub['median_cv_r2_adj'].idxmax()]
        ax.axvline(best_row['model_n_pc'], color='0.4', ls='--', lw=0.8)
        ax.set_title(f'{scope}: best PC={int(best_row["model_n_pc"])}')
        ax.set_xlabel('Number of PCs')
        ax.set_xticks(PC_GRID)
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel('Median R2_adj across cells')
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{area} all-shuffle encoding 5-fold PC grid search')
    fig.tight_layout()
    out = area_dir / f'{area}_allshuffle_5fold_pc_grid_search_r2.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


#%% 主流程
def run_all():
    _, cond_coords, pca_cache = load_or_build_global_fc6_pca(force=FORCE_FEATURE_REBUILD)
    if cond_coords.shape != (N_COND, N_PC):
        raise ValueError(f'Expected cond_coords shape {(N_COND, N_PC)}, got {cond_coords.shape}.')

    best_rows = []
    grid_rows = []
    print(
        f'Feature cache ready: condition coords={cond_coords.shape}, '
        f'first {N_PC} PCs explain '
        f'{np.sum(pca_cache["explained_variance_ratio"]):.2%} fc6 variance.'
    )

    for area in tqdm(BRAIN_AREAS, desc='area'):
        try:
            cond_rsp, cell_info, ceiling = load_area_data(area)
        except FileNotFoundError as exc:
            warnings.warn(f'Skip {area}: {exc}')
            continue

        area_grid_rows = []
        for scope in SCOPES:
            cond_idx = condition_indices(scope)
            group_ids = COND_TABLE.loc[cond_idx, 'object_id'].to_numpy(dtype=int)
            y_cond = cond_rsp[:, cond_idx]

            best_result = None
            best_row = None
            best_metric = -np.inf

            for n_model_pc in PC_GRID:
                x_cond = cond_coords[cond_idx, :n_model_pc]
                result = fit_group_model_5fold(x_cond, y_cond, group_ids, ceiling)
                row = summarize_result(area, scope, cond_idx, group_ids, n_model_pc, result)
                row['n_cell'] = cond_rsp.shape[0]
                area_grid_rows.append(row)
                grid_rows.append(row.copy())

                score = best_score(result)
                if score > best_metric:
                    best_metric = score
                    best_result = result
                    best_row = row.copy()

            npz_path, csv_path = save_group_result(
                area, scope, cond_idx, best_result, cell_info, ceiling, tag='best'
            )
            best_row['npz_path'] = str(npz_path)
            best_row['cell_metrics_csv'] = str(csv_path)
            best_rows.append(best_row)
            print(
                f'[{area} allshuffle {scope}] best PC={int(best_row["model_n_pc"])}, '
                f'median fit R2={best_row["median_fit_r2"]:.3f}, '
                f'5-fold CV R2={best_row["median_cv_r2"]:.3f}'
            )

        fig_path = plot_area_grid_search(area, area_grid_rows)
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
