r'''
使用 AlexNet fc6 建立 encoding model，并使用 leave-one-image-out 验证。

核心思想：
1. 先对全部 1000 张 Metamer1k 图片提取 AlexNet fc6 表征。
   得到的矩阵形状是 1000 × 4096（等价于“4096 × 1000”的转置）。

2. PCA 只做一次，而且是在全部 1000 张图片的 fc6 矩阵上做。
   这样得到的是所有图片共享的低维视觉表征空间，而不是每个模型单独 PCA。
   后续所有脑区、shuffle level、ani/inani/all 模型都共用这个 PCA 空间。

3. avr_rsp.npy 的列顺序按：
   5 repeat × (5 shuffle level × 40 image)
   本脚本先平均同一 condition 的 5 个 repeat，得到 200 个 condition-level 响应。

4. 本脚本不再分别拟合每个 shuffle level，而是合并全部 5 个 shuffle level：
   ani: 20 个 object × 5 shuffle level = 100 个 condition
   inani: 20 个 object × 5 shuffle level = 100 个 condition
   all: 40 个 object × 5 shuffle level = 200 个 condition

5. 对每个脑区、每个图片组分别拟合线性 encoding model：
   PCA 图片表征 -> 每个神经元的 condition-level 平均响应。
   所有神经元共享同一个输入矩阵，但每个神经元有自己独立的线性权重。

6. 评估指标：
   fit_r2: 用全部 condition 拟合后，在这些 condition 上的 R2。
   loo_r2: leave-one-object-out 验证 R2。
           每次留出一个 object 的全部 5 个 shuffle-level condition。
   R2_adj = R2 / ceiling_index，其中 ceiling_index 来自 cell_site_info.csv。

7. 保存内容：
   对每个 area / ani-inani-all 组合，grid search PC 数目 1-20。
   保存完整 grid search 表、fit/LOO R2 曲线图，以及 LOO R2 最好的模型。

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
BATCH_SIZE = 32

WINDOW_S = 0.17
CONVERT_TO_HZ = False
FORCE_FEATURE_REBUILD = False

GLOBAL_PCA_CACHE = SAVEPATH / 'alexnet_metamer1k_global_fc6_pca.npz'
SUMMARY_CSV = SAVEPATH / 'encoding_model_fc6_summary.csv'
GRID_SEARCH_CSV = SAVEPATH / 'encoding_model_fc6_pc_grid_search.csv'

SAVEPATH.mkdir(parents=True, exist_ok=True)


#%% 刺激索引
def condition_table():
    """Return the 200 unique image conditions shared by the 5 repeat cycles."""
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
        raise ValueError(
            f'scope={scope}: expected {expected} '
            f'conditions, got {len(idx)}.'
        )
    return idx


#%% 全部 1000 张图片的 AlexNet fc6 + 全局 PCA
def image_paths_metamer1k():
    paths = [STIMPATH / f'{i:04d}.jpg' for i in range(1, N_METAMER + 1)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Missing metamer image files, first missing file: {missing[0]}'
        )
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
        # classifier[2] is an inplace ReLU, so clone fc6 immediately.
        buf = []

        def _hook(_module, _inp, out):
            buf.append(out.detach().cpu().clone())

        handle = model.classifier[1].register_forward_hook(_hook)
        model(batch_tensor)
        handle.remove()
        return buf[0]

    sample = _extract(preprocess(Image.open(img_paths[0]).convert('RGB')).unsqueeze(0).to(device))
    feat_dim = int(sample.shape[1])
    fc6 = np.zeros((len(img_paths), feat_dim), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, len(img_paths), BATCH_SIZE), desc='AlexNet fc6'):
            batch_paths = img_paths[start:start + BATCH_SIZE]
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
            fc6[start:start + len(imgs)] = _extract(torch.stack(imgs).to(device)).numpy()
    return fc6


def load_or_build_global_fc6_pca(force=False):
    """
    Build one common PCA space from all 1000 Metamer1k images.

    This is the only PCA step in the whole script. Individual encoding models
    only select rows from this shared PCA representation.
    """
    if GLOBAL_PCA_CACHE.is_file() and not force:
        d = np.load(GLOBAL_PCA_CACHE, allow_pickle=True)
        if int(d['n_pc']) != N_PC:
            if 'fc6' not in d.files:
                raise ValueError(
                    f'{GLOBAL_PCA_CACHE} was built with n_pc={int(d["n_pc"])}, '
                    f'but current N_PC={N_PC}, and no cached fc6 is available.'
                )
            warnings.warn(
                f'{GLOBAL_PCA_CACHE} was built with n_pc={int(d["n_pc"])}; '
                f'rebuilding global PCA with n_pc={N_PC} from cached fc6.'
            )
            fc6 = d['fc6']
            img_paths = d['img_paths'] if 'img_paths' in d.files else np.array([], dtype=object)
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
                img_paths=img_paths,
            )
            d = np.load(GLOBAL_PCA_CACHE, allow_pickle=True)
        print(f'Loaded global fc6 PCA cache: {GLOBAL_PCA_CACHE}')
        coords = d['coords']
        if 'cond_coords' in d.files:
            cond_coords = d['cond_coords']
        else:
            cond_coords = coords.reshape(N_REPEAT, N_COND, N_PC).mean(axis=0).astype(np.float32)
        return coords, cond_coords, d

    from sklearn.decomposition import PCA

    img_paths = image_paths_metamer1k()
    fc6 = extract_fc6_features(img_paths)

    # fc6 shape: 1000 images × 4096 features.
    # PCA is fitted once on all images to capture shared image structure.
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
    print(f'Saved global fc6 PCA cache: {GLOBAL_PCA_CACHE}')
    return coords, cond_coords, np.load(GLOBAL_PCA_CACHE, allow_pickle=True)


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
            f'{area}: expected avr_rsp columns {N_METAMER} or {N_COND}, '
            f'got {rsp.shape[1]}.'
        )

    ceiling = pd.to_numeric(cell_info['ceiling_index'], errors='coerce').to_numpy(np.float64)
    return cond_rsp, cell_info, ceiling


#%% 线性模型与评估
def standardize_train_test(x_train, x_test=None):
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x_train_z = (x_train - mu) / sd
    if x_test is None:
        return x_train_z, mu, sd
    x_test_z = (x_test - mu) / sd
    return x_train_z, x_test_z, mu, sd


def fit_linear_multioutput(x, y):
    x_aug = np.c_[x, np.ones(len(x))]
    coef_aug, _, _, _ = np.linalg.lstsq(x_aug, y.T, rcond=None)
    weights = coef_aug[:-1].astype(np.float32)
    bias = coef_aug[-1].astype(np.float32)
    pred = x_aug @ coef_aug
    return weights, bias, pred


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


def fit_group_model(x_cond, y_cond, group_ids, ceiling):
    """Fit all cells independently, sharing the same feature matrix."""
    x_z, x_mean, x_std = standardize_train_test(x_cond)
    weights, bias, y_fit = fit_linear_multioutput(x_z, y_cond)
    fit_r2 = r2_per_cell(y_cond.T, y_fit)

    y_loo = np.full_like(y_cond.T, np.nan, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    for heldout_group in np.unique(group_ids):
        test_mask = group_ids == heldout_group
        train_mask = ~test_mask
        x_tr, x_te, _, _ = standardize_train_test(x_cond[train_mask], x_cond[test_mask])
        x_tr_aug = np.c_[x_tr, np.ones(len(x_tr))]
        coef_aug, _, _, _ = np.linalg.lstsq(x_tr_aug, y_cond[:, train_mask].T, rcond=None)
        pred_te = np.c_[x_te, np.ones(len(x_te))] @ coef_aug
        y_loo[test_mask] = pred_te

    loo_r2 = r2_per_cell(y_cond.T, y_loo)
    return {
        'weights': weights,
        'bias': bias,
        'x_mean': x_mean.astype(np.float32),
        'x_std': x_std.astype(np.float32),
        'y_fit': y_fit.T.astype(np.float32),
        'y_loo': y_loo.T.astype(np.float32),
        'fit_r2': fit_r2,
        'loo_r2': loo_r2,
        'fit_r2_adj': adjust_r2_by_ceiling(fit_r2, ceiling),
        'loo_r2_adj': adjust_r2_by_ceiling(loo_r2, ceiling),
    }


def save_group_result(area, scope, cond_idx, result, cell_info, ceiling, tag='best'):
    area_dir = SAVEPATH / area
    area_dir.mkdir(parents=True, exist_ok=True)
    n_model_pc = int(result['weights'].shape[0])
    stem = f'{area}_allshuffle_{scope}_{tag}_fc6pc{n_model_pc}'
    npz_path = area_dir / f'{stem}.npz'
    csv_path = area_dir / f'{stem}_cell_metrics.csv'

    np.savez(
        npz_path,
        weights=result['weights'],
        bias=result['bias'],
        x_mean=result['x_mean'],
        x_std=result['x_std'],
        y_fit=result['y_fit'],
        y_loo=result['y_loo'],
        fit_r2=result['fit_r2'],
        loo_r2=result['loo_r2'],
        fit_r2_adj=result['fit_r2_adj'],
        loo_r2_adj=result['loo_r2_adj'],
        ceiling_index=ceiling.astype(np.float32),
        cond_idx=cond_idx.astype(np.int32),
        object_id=COND_TABLE.loc[cond_idx, 'object_id'].to_numpy(dtype=np.int32),
        shuffle=COND_TABLE.loc[cond_idx, 'shuffle'].to_numpy(dtype=np.int32),
        shuffle_label=COND_TABLE.loc[cond_idx, 'shuffle_label'].to_numpy(dtype=object),
        scope=np.array(scope),
        feature_n_pc=np.array(N_PC),
        model_n_pc=np.array(n_model_pc),
    )

    metrics = pd.DataFrame({
        'cell_idx': np.arange(len(ceiling)),
        'ceiling_index': ceiling,
        'fit_r2': result['fit_r2'],
        'fit_r2_adj': result['fit_r2_adj'],
        'loo_r2': result['loo_r2'],
        'loo_r2_adj': result['loo_r2_adj'],
    })
    passthrough_cols = [
        col for col in ('cell_id', 'site', 'area', 'dprime_face', 'dprime_body')
        if col in cell_info.columns
    ]
    if passthrough_cols:
        metrics = pd.concat(
            [cell_info[passthrough_cols].reset_index(drop=True), metrics],
            axis=1,
        )
    metrics.to_csv(csv_path, index=False)
    return npz_path, csv_path


def summarize_result(area, scope, cond_idx, group_ids, n_model_pc, result):
    return {
        'area': area,
        'scope': scope,
        'n_condition': len(cond_idx),
        'n_loo_groups': len(np.unique(group_ids)),
        'loo_group': 'object_id',
        'n_shuffle': N_SHUF,
        'n_repeat': N_REPEAT,
        'feature_n_pc': N_PC,
        'model_n_pc': n_model_pc,
        'median_fit_r2': np.nanmedian(result['fit_r2']),
        'median_fit_r2_adj': np.nanmedian(result['fit_r2_adj']),
        'median_loo_r2': np.nanmedian(result['loo_r2']),
        'median_loo_r2_adj': np.nanmedian(result['loo_r2_adj']),
        'mean_fit_r2': np.nanmean(result['fit_r2']),
        'mean_fit_r2_adj': np.nanmean(result['fit_r2_adj']),
        'mean_loo_r2': np.nanmean(result['loo_r2']),
        'mean_loo_r2_adj': np.nanmean(result['loo_r2_adj']),
    }


def best_score(result):
    score = float(np.nanmedian(result['loo_r2_adj']))
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
        ax.plot(sub['model_n_pc'], sub['median_loo_r2_adj'], '-o', ms=3, label='LOO R2_adj')
        best_row = sub.loc[sub['median_loo_r2_adj'].idxmax()]
        ax.axvline(best_row['model_n_pc'], color='0.4', ls='--', lw=0.8)
        ax.set_title(f'{scope}: best PC={int(best_row["model_n_pc"])}')
        ax.set_xlabel('Number of PCs')
        ax.set_xticks(PC_GRID)
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel('Median R2_adj across cells')
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{area} all-shuffle encoding PC grid search')
    fig.tight_layout()
    out = area_dir / f'{area}_allshuffle_pc_grid_search_r2.png'
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
                result = fit_group_model(x_cond, y_cond, group_ids, ceiling)
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
                f'LOO R2={best_row["median_loo_r2"]:.3f}'
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


