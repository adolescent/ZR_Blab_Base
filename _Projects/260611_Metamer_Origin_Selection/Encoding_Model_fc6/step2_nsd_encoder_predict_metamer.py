r'''
使用 step1 中 NSD 数据训练出的 encoder，评估其对 Metamer1k 响应的预测效果。

输入：
1. NSD encoder 最佳模型：
   E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Encoder_fc6\encoding_model_nsd_summary.csv

2. NSD fc6 PCA 缓存：
   E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Encoder_fc6\alexnet_nsd1k_global_fc6_pca.npz

3. Metamer_NSD_2k 响应：
   每个脑区的 avr_rsp.npy 前 1000 列是 Metamer1k 响应。

输出：
1. 每个脑区的预测矩阵和逐细胞 R2/R2_adj。
2. 总 summary：整体、按 5 个 shuffle level 的 R2/R2_adj。
3. 每个脑区一张预测效果分布图。

保存目录：
E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Encoder_fc6\metamer_predict
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
OUTPATH = SAVEPATH / 'metamer_predict'
DATAPATH = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_NSD_2k')
METAMER_STIMPATH = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Stimsets\Metamer1k')

NSD_SUMMARY_CSV = SAVEPATH / 'encoding_model_nsd_summary.csv'
NSD_PCA_CACHE = SAVEPATH / 'alexnet_nsd1k_global_fc6_pca.npz'
METAMER_FEATURE_CACHE = OUTPATH / 'alexnet_metamer1k_in_nsd_pca_space.npz'

N_METAMER = 1000
N_SHUF = 5
N_OBJ = 40
N_COND_PER_CYCLE = N_SHUF * N_OBJ
BATCH_SIZE = 32
WINDOW_S = 0.17
CONVERT_TO_HZ = False
FORCE_METAMER_FEATURE_REBUILD = False

OUTPATH.mkdir(parents=True, exist_ok=True)


#%% stimulus labels
def metamer_table():
    idx = np.arange(N_METAMER)
    within = idx % N_COND_PER_CYCLE
    return pd.DataFrame({
        'stim_idx': idx,
        'cycle': idx // N_COND_PER_CYCLE,
        'shuffle': within // N_OBJ,
        'shuffle_label': np.array(['Raw', 'S1', 'S2', 'S3', 'S4'])[within // N_OBJ],
        'object_id': within % N_OBJ,
    })


STIM_TABLE = metamer_table()


def eval_masks():
    masks = {'overall': np.ones(N_METAMER, dtype=bool)}
    for shuf in range(N_SHUF):
        masks[f'shuffle_{shuf}'] = STIM_TABLE['shuffle'].to_numpy() == shuf
    return masks


#%% feature extraction and PCA projection
def metamer_image_paths():
    paths = [METAMER_STIMPATH / f'{i:04d}.jpg' for i in range(1, N_METAMER + 1)]
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
            # classifier[2] is inplace ReLU, so clone fc6 immediately.
            buf.append(out.detach().cpu().clone())

        handle = model.classifier[1].register_forward_hook(_hook)
        model(batch_tensor)
        handle.remove()
        return buf[0]

    sample = _extract(preprocess(Image.open(img_paths[0]).convert('RGB')).unsqueeze(0).to(device))
    fc6 = np.zeros((len(img_paths), int(sample.shape[1])), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, len(img_paths), BATCH_SIZE), desc='AlexNet fc6 Metamer'):
            batch_paths = img_paths[start:start + BATCH_SIZE]
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
            fc6[start:start + len(imgs)] = _extract(torch.stack(imgs).to(device)).numpy()
    return fc6


def load_or_build_metamer_pca_coords(force=False):
    if METAMER_FEATURE_CACHE.is_file() and not force:
        d = np.load(METAMER_FEATURE_CACHE, allow_pickle=True)
        print(f'Loaded metamer feature cache: {METAMER_FEATURE_CACHE}')
        return d['coords'].astype(np.float32), d

    if not NSD_PCA_CACHE.is_file():
        raise FileNotFoundError(
            f'Missing NSD PCA cache: {NSD_PCA_CACHE}. Run step1_encoding_model_nsd.py first.'
        )
    nsd_pca = np.load(NSD_PCA_CACHE, allow_pickle=True)
    for key in ('pca_mean', 'pca_components'):
        if key not in nsd_pca.files:
            raise KeyError(f'{NSD_PCA_CACHE} lacks {key}; rerun step1 to rebuild NSD PCA from fc6.')

    img_paths = metamer_image_paths()
    fc6 = extract_fc6_features(img_paths)
    coords = ((fc6 - nsd_pca['pca_mean']) @ nsd_pca['pca_components'].T).astype(np.float32)
    np.savez(
        METAMER_FEATURE_CACHE,
        fc6=fc6,
        coords=coords,
        img_paths=np.array([str(p) for p in img_paths], dtype=object),
        nsd_pca_cache=np.array(str(NSD_PCA_CACHE)),
    )
    print(f'Saved metamer feature cache: {METAMER_FEATURE_CACHE}')
    return coords, np.load(METAMER_FEATURE_CACHE, allow_pickle=True)


#%% response and metrics
def metamer_slice():
    layout_path = DATAPATH / 'stim_layout.npz'
    if not layout_path.is_file():
        warnings.warn('stim_layout.npz not found; using first 1000 columns as metamer responses.')
        return slice(0, N_METAMER)

    layout = np.load(layout_path, allow_pickle=True)
    if 'slice_metamer' in layout.files:
        values = np.asarray(layout['slice_metamer'], dtype=int).ravel()
        return slice(int(values[0]), int(values[1]))
    return slice(0, N_METAMER)


def load_area_metamer_response(area, response_slice):
    area_path = DATAPATH / area
    rsp_all = np.load(area_path / 'avr_rsp.npy').astype(np.float64)
    cell_info = pd.read_csv(area_path / 'cell_site_info.csv')

    if rsp_all.shape[1] < response_slice.stop:
        raise ValueError(f'{area}: expected at least {response_slice.stop} columns, got {rsp_all.shape[1]}.')
    if len(cell_info) != rsp_all.shape[0]:
        raise ValueError(
            f'{area}: cell_site_info rows ({len(cell_info)}) do not match response cells ({rsp_all.shape[0]}).'
        )
    if 'ceiling_index' not in cell_info.columns:
        raise KeyError(f'{area}: cell_site_info.csv lacks column "ceiling_index".')

    rsp = rsp_all[:, response_slice].astype(np.float64)
    if rsp.shape[1] != N_METAMER:
        raise ValueError(f'{area}: expected {N_METAMER} metamer responses, got {rsp.shape[1]}.')
    if CONVERT_TO_HZ:
        rsp = rsp / WINDOW_S

    ceiling = pd.to_numeric(cell_info['ceiling_index'], errors='coerce').to_numpy(np.float64)
    return rsp, cell_info, ceiling


def r2_per_cell(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan).astype(np.float32)


def adjust_r2_by_ceiling(r2, ceiling):
    out = np.full_like(r2, np.nan, dtype=np.float32)
    valid = np.isfinite(r2) & np.isfinite(ceiling) & (ceiling > 0)
    out[valid] = (r2[valid] / ceiling[valid]).astype(np.float32)
    return out


def predict_from_encoder(model, metamer_coords):
    model_n_pc = int(model['model_n_pc'])
    x = metamer_coords[:, :model_n_pc]
    x_z = (x - model['x_mean']) / model['x_std']
    pred = x_z @ model['weights'] + model['bias']
    return pred.T.astype(np.float32)


def metric_rows(area, r2_by_group, r2_adj_by_group, n_cell, model_n_pc, model_path):
    rows = []
    for group_name in r2_by_group:
        rows.append({
            'area': area,
            'group': group_name,
            'model_n_pc': model_n_pc,
            'n_cell': n_cell,
            'median_r2': np.nanmedian(r2_by_group[group_name]),
            'median_r2_adj': np.nanmedian(r2_adj_by_group[group_name]),
            'mean_r2': np.nanmean(r2_by_group[group_name]),
            'mean_r2_adj': np.nanmean(r2_adj_by_group[group_name]),
            'model_path': str(model_path),
        })
    return rows


def plot_area_summary(area, rows_df, cell_metrics):
    import matplotlib.pyplot as plt

    area_dir = OUTPATH / area
    area_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), dpi=150)
    groups = ['overall', 'shuffle_0', 'shuffle_1', 'shuffle_2', 'shuffle_3', 'shuffle_4']
    sub = rows_df.set_index('group').loc[groups]
    x = np.arange(len(groups))
    axes[0].bar(x - 0.18, sub['median_r2'], width=0.36, label='R2')
    axes[0].bar(x + 0.18, sub['median_r2_adj'], width=0.36, label='R2_adj')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups, rotation=45, ha='right', fontsize=7)
    axes[0].set_ylabel('Median across cells')
    axes[0].set_title(f'{area}: group performance')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis='y', alpha=0.25, lw=0.5)

    valid = cell_metrics['r2_adj'].to_numpy()
    valid = valid[np.isfinite(valid)]
    axes[1].hist(valid, bins=40, color='0.45', edgecolor='white', lw=0.5)
    axes[1].axvline(np.nanmedian(valid), color='C1', ls='--', lw=1.2, label='median')
    axes[1].set_xlabel('Overall R2_adj')
    axes[1].set_ylabel('Cell count')
    axes[1].set_title('Cell distribution')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out = area_dir / f'{area}_nsd_encoder_predict_metamer_summary.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


#%% main
def run_all():
    if not NSD_SUMMARY_CSV.is_file():
        raise FileNotFoundError(f'Missing NSD encoder summary: {NSD_SUMMARY_CSV}')

    nsd_summary = pd.read_csv(NSD_SUMMARY_CSV)
    metamer_coords, _ = load_or_build_metamer_pca_coords(force=FORCE_METAMER_FEATURE_REBUILD)
    response_slice = metamer_slice()
    masks = eval_masks()

    all_rows = []
    for _, model_row in tqdm(nsd_summary.iterrows(), total=len(nsd_summary), desc='area'):
        area = model_row['area']
        model_path = Path(model_row['npz_path'])
        if not model_path.is_file():
            warnings.warn(f'Skip {area}: model file not found: {model_path}')
            continue

        model = np.load(model_path, allow_pickle=True)
        true_rsp, cell_info, ceiling = load_area_metamer_response(area, response_slice)
        pred_rsp = predict_from_encoder(model, metamer_coords)
        if pred_rsp.shape != true_rsp.shape:
            raise ValueError(f'{area}: prediction shape {pred_rsp.shape} != response shape {true_rsp.shape}.')

        r2_by_group = {}
        r2_adj_by_group = {}
        for group_name, mask in masks.items():
            r2 = r2_per_cell(true_rsp[:, mask], pred_rsp[:, mask])
            r2_by_group[group_name] = r2
            r2_adj_by_group[group_name] = adjust_r2_by_ceiling(r2, ceiling)

        cell_metrics = pd.DataFrame({
            'cell_idx': np.arange(true_rsp.shape[0]),
            'ceiling_index': ceiling,
            'r2': r2_by_group['overall'],
            'r2_adj': r2_adj_by_group['overall'],
        })
        for shuf in range(N_SHUF):
            cell_metrics[f'r2_shuffle_{shuf}'] = r2_by_group[f'shuffle_{shuf}']
            cell_metrics[f'r2_adj_shuffle_{shuf}'] = r2_adj_by_group[f'shuffle_{shuf}']

        passthrough_cols = [
            col for col in ('cell_id', 'site', 'area', 'dprime_face', 'dprime_body')
            if col in cell_info.columns
        ]
        if passthrough_cols:
            cell_metrics = pd.concat([cell_info[passthrough_cols].reset_index(drop=True), cell_metrics], axis=1)

        area_dir = OUTPATH / area
        area_dir.mkdir(parents=True, exist_ok=True)
        pred_path = area_dir / f'{area}_nsd_encoder_metamer_prediction.npz'
        cell_csv = area_dir / f'{area}_nsd_encoder_metamer_cell_metrics.csv'

        np.savez(
            pred_path,
            pred_rsp=pred_rsp.astype(np.float32),
            true_rsp=true_rsp.astype(np.float32),
            metamer_coords=metamer_coords.astype(np.float32),
            ceiling_index=ceiling.astype(np.float32),
            stim_idx=STIM_TABLE['stim_idx'].to_numpy(dtype=np.int32),
            cycle=STIM_TABLE['cycle'].to_numpy(dtype=np.int32),
            object_id=STIM_TABLE['object_id'].to_numpy(dtype=np.int32),
            shuffle=STIM_TABLE['shuffle'].to_numpy(dtype=np.int32),
            shuffle_label=STIM_TABLE['shuffle_label'].to_numpy(dtype=object),
            nsd_model_path=np.array(str(model_path)),
            model_n_pc=np.array(int(model['model_n_pc'])),
        )
        cell_metrics.to_csv(cell_csv, index=False)

        rows = metric_rows(
            area,
            r2_by_group,
            r2_adj_by_group,
            true_rsp.shape[0],
            int(model['model_n_pc']),
            model_path,
        )
        rows_df = pd.DataFrame(rows)
        fig_path = plot_area_summary(area, rows_df, cell_metrics)
        for row in rows:
            row['prediction_npz'] = str(pred_path)
            row['cell_metrics_csv'] = str(cell_csv)
            row['figure_path'] = str(fig_path)
        all_rows.extend(rows)

        overall = rows_df[rows_df['group'] == 'overall'].iloc[0]
        print(
            f'[{area}] Metamer prediction: median R2={overall["median_r2"]:.3f}, '
            f'R2_adj={overall["median_r2_adj"]:.3f}, model PC={int(model["model_n_pc"])}'
        )

    summary = pd.DataFrame(all_rows)
    summary_csv = OUTPATH / 'nsd_encoder_predict_metamer_summary.csv'
    summary.to_csv(summary_csv, index=False)
    print(f'Saved summary: {summary_csv}')
    return summary


if __name__ == '__main__':
    run_all()


#%% 一些可视化

def plot_shuffle_r2_heatmaps(metric='r2', sort_cells=True):
    """Rows=cells, columns=5 shuffle levels; metric is 'r2' or 'r2_adj'."""
    import matplotlib.pyplot as plt

    rows = pd.read_csv(OUTPATH / 'nsd_encoder_predict_metamer_summary.csv')
    rows = rows[rows['group'] == 'overall']
    shuffle_cols = [f'{metric}_shuffle_{shuf}' for shuf in range(N_SHUF)]
    out_files = []

    for _, row in rows.iterrows():
        area = row['area']
        data = pd.read_csv(row['cell_metrics_csv'])[shuffle_cols].to_numpy(float)
        if sort_cells:
            data = data[np.argsort(np.nanmean(data, axis=1))[::-1]]

        vmax = 1
        fig, ax = plt.subplots(figsize=(4.5, max(4, min(12, data.shape[0] / 120))), dpi=150)
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(N_SHUF))
        ax.set_xticklabels(['Raw', 'S1', 'S2', 'S3', 'S4'])
        ax.set_xlabel('Shuffle level')
        ax.set_ylabel('Neuron')
        ax.set_title(f'{area}: {metric} by shuffle level')
        fig.colorbar(im, ax=ax, label=metric)
        fig.tight_layout()
        out = OUTPATH / area / f'{area}_shuffle_level_cell_{metric}_heatmap.png'
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        out_files.append(out)
        print(f'[{area}] saved shuffle-level cell heatmap: {out}')

    return out_files


# Run this cell after run_all() if needed.
plot_shuffle_r2_heatmaps(metric='r2')
plot_shuffle_r2_heatmaps(metric='r2_adj')
#%%

def eval_shuffle_std_range():
    """For each cell/object, compare recorded vs predicted std/range across 5 shuffles."""
    import matplotlib.pyplot as plt

    summary = pd.read_csv(OUTPATH / 'nsd_encoder_predict_metamer_summary.csv')
    summary = summary[summary['group'] == 'overall']
    all_detail, all_cell, all_raw = [], [], []

    for _, row in summary.iterrows():
        area = row['area']
        d = np.load(row['prediction_npz'], allow_pickle=True)
        true, pred = d['true_rsp'], d['pred_rsp']
        obj, shuf = d['object_id'], d['shuffle']

        rows = []
        for cell in range(true.shape[0]):
            for o in range(N_OBJ):
                y = np.array([np.nanmean(true[cell, (obj == o) & (shuf == s)]) for s in range(N_SHUF)])
                p = np.array([np.nanmean(pred[cell, (obj == o) & (shuf == s)]) for s in range(N_SHUF)])
                rows.append({
                    'area': area,
                    'cell_idx': cell,
                    'object_id': o,
                    'true_std': np.nanstd(y),
                    'pred_std': np.nanstd(p),
                    'std_diff': np.nanstd(y) - np.nanstd(p),
                    'true_range': np.nanmax(y) - np.nanmin(y),
                    'pred_range': np.nanmax(p) - np.nanmin(p),
                    'range_diff': (np.nanmax(y) - np.nanmin(y)) - (np.nanmax(p) - np.nanmin(p)),
                })

        detail = pd.DataFrame(rows)
        detail['std_compressed'] = detail['pred_std'] < detail['true_std']
        detail['range_compressed'] = detail['pred_range'] < detail['true_range']
        detail['std_ratio'] = detail['pred_std'] / detail['true_std'].replace(0, np.nan)
        detail['range_ratio'] = detail['pred_range'] / detail['true_range'].replace(0, np.nan)

        cell_summary = detail.groupby(['area', 'cell_idx'], as_index=False).agg(
            mean_true_std=('true_std', 'mean'),
            mean_pred_std=('pred_std', 'mean'),
            mean_std_diff=('std_diff', 'mean'),
            frac_std_compressed=('std_compressed', 'mean'),
            mean_true_range=('true_range', 'mean'),
            mean_pred_range=('pred_range', 'mean'),
            mean_range_diff=('range_diff', 'mean'),
            frac_range_compressed=('range_compressed', 'mean'),
        )

        raw_rows = []
        for cell in range(true.shape[0]):
            m = shuf == 0
            y_raw = np.array([np.nanmean(true[cell, m & (obj == o)]) for o in range(N_OBJ)])
            p_raw = np.array([np.nanmean(pred[cell, m & (obj == o)]) for o in range(N_OBJ)])
            true_std = np.nanstd(y_raw)
            pred_std = np.nanstd(p_raw)
            true_range = np.nanmax(y_raw) - np.nanmin(y_raw)
            pred_range = np.nanmax(p_raw) - np.nanmin(p_raw)
            raw_rows.append({
                'area': area,
                'cell_idx': cell,
                'raw_true_std_across_object': true_std,
                'raw_pred_std_across_object': pred_std,
                'raw_std_diff': true_std - pred_std,
                'raw_std_compressed': pred_std < true_std,
                'raw_true_range_across_object': true_range,
                'raw_pred_range_across_object': pred_range,
                'raw_range_diff': true_range - pred_range,
                'raw_range_compressed': pred_range < true_range,
            })
        raw_summary = pd.DataFrame(raw_rows)

        area_dir = OUTPATH / area
        detail_csv = area_dir / f'{area}_shuffle_std_range_by_cell_object.csv'
        cell_csv = area_dir / f'{area}_shuffle_std_range_by_cell.csv'
        raw_csv = area_dir / f'{area}_raw_object_std_range_by_cell.csv'
        detail.to_csv(detail_csv, index=False)
        cell_summary.to_csv(cell_csv, index=False)
        raw_summary.to_csv(raw_csv, index=False)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=150)
        axes[0].scatter(detail['true_std'], detail['pred_std'], s=4, alpha=0.25)
        axes[1].scatter(detail['true_range'], detail['pred_range'], s=4, alpha=0.25)
        for ax, name in zip(axes, ['std', 'range']):
            lim = ax.get_xlim() + ax.get_ylim()
            lim = (min(lim), max(lim))
            ax.plot(lim, lim, 'k--', lw=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(f'recorded {name}')
            ax.set_ylabel(f'predicted {name}')
            ax.set_title(f'{area}: shuffle {name}')
        fig.tight_layout()
        fig_path = area_dir / f'{area}_shuffle_std_range_pred_vs_true.png'
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3), dpi=150)
        axes[0].scatter(raw_summary['raw_true_std_across_object'], raw_summary['raw_pred_std_across_object'], s=8, alpha=0.45)
        axes[1].scatter(raw_summary['raw_true_range_across_object'], raw_summary['raw_pred_range_across_object'], s=8, alpha=0.45)
        for ax, name in zip(axes, ['raw object std', 'raw object range']):
            lim = ax.get_xlim() + ax.get_ylim()
            lim = (min(lim), max(lim))
            ax.plot(lim, lim, 'k--', lw=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(f'recorded {name}')
            ax.set_ylabel(f'predicted {name}')
            ax.set_title(area)
        fig.tight_layout()
        raw_fig_path = area_dir / f'{area}_raw_object_std_range_pred_vs_true.png'
        fig.savefig(raw_fig_path, bbox_inches='tight')
        plt.close(fig)

        all_detail.append(detail)
        all_cell.append(cell_summary)
        all_raw.append(raw_summary)
        print(f'[{area}] saved std/range stats: {detail_csv}, {cell_csv}, {raw_csv}')

    pd.concat(all_detail).to_csv(OUTPATH / 'shuffle_std_range_by_cell_object.csv', index=False)
    pd.concat(all_cell).to_csv(OUTPATH / 'shuffle_std_range_by_cell.csv', index=False)
    pd.concat(all_raw).to_csv(OUTPATH / 'raw_object_std_range_by_cell.csv', index=False)


eval_shuffle_std_range()
#%%
def plot_demo_cell_prediction(area='ML', cell_idx=0):
    """
    Demo for one manually selected neuron.

    The 1000 metamer images are averaged into 200 conditions:
    5 repeat cycles × (5 shuffle levels × 40 objects) -> 5 shuffle × 40 objects.
    """
    import matplotlib.pyplot as plt

    summary_csv = OUTPATH / 'nsd_encoder_predict_metamer_summary.csv'
    if not summary_csv.is_file():
        raise FileNotFoundError(f'Missing {summary_csv}. Run run_all() first.')

    summary = pd.read_csv(summary_csv)
    hits = summary[(summary['area'] == area) & (summary['group'] == 'overall')]
    if hits.empty:
        areas = sorted(summary['area'].unique())
        raise ValueError(f'Area {area!r} not found. Available areas: {areas}')

    pred_file = Path(hits.iloc[0]['prediction_npz'])
    metrics_file = Path(hits.iloc[0]['cell_metrics_csv'])
    if not pred_file.is_file() or not metrics_file.is_file():
        raise FileNotFoundError(f'Missing prediction files for {area}. Run run_all() again.')

    d = np.load(pred_file, allow_pickle=True)
    metrics = pd.read_csv(metrics_file)

    true = d['true_rsp'][cell_idx]
    pred = d['pred_rsp'][cell_idx]
    shuf = d['shuffle']
    obj = d['object_id']

    true_mat = np.full((N_SHUF, N_OBJ), np.nan)
    pred_mat = np.full((N_SHUF, N_OBJ), np.nan)
    for s in range(N_SHUF):
        for o in range(N_OBJ):
            m = (shuf == s) & (obj == o)
            true_mat[s, o] = np.nanmean(true[m])
            pred_mat[s, o] = np.nanmean(pred[m])

    resid_mat = pred_mat - true_mat
    true_range = np.nanmax(true_mat, axis=0) - np.nanmin(true_mat, axis=0)
    pred_range = np.nanmax(pred_mat, axis=0) - np.nanmin(pred_mat, axis=0)

    vmin = np.nanpercentile(np.r_[true_mat, pred_mat], 2)
    vmax = np.nanpercentile(np.r_[true_mat, pred_mat], 98)
    err_max = max(np.nanpercentile(np.abs(resid_mat), 98), 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150)
    for ax, mat, title in [
        (axes[0, 0], true_mat, 'Recorded response'),
        (axes[0, 1], pred_mat, 'Predicted response'),
    ]:
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Object id')
        ax.set_ylabel('Shuffle level')
        ax.set_yticks(range(N_SHUF))
        ax.set_yticklabels(['Raw', 'S1', 'S2', 'S3', 'S4'])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(resid_mat, aspect='auto', cmap='RdBu_r', vmin=-err_max, vmax=err_max)
    axes[1, 0].set_title('Prediction error (pred - recorded)')
    axes[1, 0].set_xlabel('Object id')
    axes[1, 0].set_ylabel('Shuffle level')
    axes[1, 0].set_yticks(range(N_SHUF))
    axes[1, 0].set_yticklabels(['Raw', 'S1', 'S2', 'S3', 'S4'])
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    for o in range(N_OBJ):
        ax.plot([0, 1], [true_range[o], pred_range[o]], color='0.75', lw=0.7)
    ax.scatter(np.zeros(N_OBJ), true_range, s=16, label='recorded')
    ax.scatter(np.ones(N_OBJ), pred_range, s=16, label='predicted')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['recorded', 'predicted'])
    ax.set_ylabel('Range across 5 shuffle levels')
    ax.set_title('Shuffle modulation by object')
    ax.legend(frameon=False, fontsize=8)

    r2 = metrics.loc[cell_idx, 'r2']
    r2_adj = metrics.loc[cell_idx, 'r2_adj']
    fig.suptitle(f'{area} cell {cell_idx}: metamer prediction  R2={r2:.3f}, R2_adj={r2_adj:.3f}')
    fig.tight_layout()

    out = OUTPATH / area / f'{area}_cell{cell_idx}_metamer_prediction_demo.png'
    fig.savefig(out, bbox_inches='tight')
    plt.show()
    print(f'Saved demo figure: {out}')
    return out


# 手动修改这里，取消注释后运行：
plot_demo_cell_prediction(area='ASB', cell_idx=83)

#%%



