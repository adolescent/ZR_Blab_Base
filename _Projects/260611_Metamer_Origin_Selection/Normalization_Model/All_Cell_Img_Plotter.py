'''
展示每个神经元对每幅图的激活野和抑制野。

在图上标注 bubble 拟合 R2、完整图片预测 R2，以及全部模型参数。
以 alpha=0.7 的方式展示激活和抑制野，叠加在原始图片上。
'''

#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---- paths and config ----
fit_savepath = Path(
    r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Normalization_Bubble'
)
fig_savepath = Path(
    r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Normalization_Bubble\All_Cell_Demo'
)
raw_img_path = Path(
    r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Stimsets\Raw_Objs'
)
datapath = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_SingleBubble')

AREAS = ['ML', 'MSB', 'AL', 'ASB']
OVERLAY_ALPHA = 0.7
DPI = 150

# 每个脑区随机抽取的神经元数；None 表示全部。
N_CELLS_PER_AREA = 100
SAMPLE_SEED = 12580
MAX_OBJECTS = None
SKIP_EXISTING = True


#%%
def sample_cell_ids(cell_ids, n_sample, rng):
    cell_ids = sorted(int(x) for x in cell_ids)
    if n_sample is None or n_sample >= len(cell_ids):
        return cell_ids
    picked = rng.choice(cell_ids, size=n_sample, replace=False)
    return sorted(int(x) for x in picked)


def gaussian_map(xx, yy, x0, y0, std):
    std = max(float(std), 1e-6)
    g = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / std ** 2)
    total = g.sum()
    if total <= 0:
        return np.full_like(g, 1.0 / g.size)
    return g / total


def load_raw_pred_cell_df():
    pred_cell_pkl = fit_savepath / 'normalization_bubble_raw_pred_cell.pkl'
    if pred_cell_pkl.exists():
        return pd.read_pickle(pred_cell_pkl)

    pred_pkl = fit_savepath / 'normalization_bubble_raw_pred.pkl'
    if not pred_pkl.exists():
        return None

    pred_df = pd.read_pickle(pred_pkl)
    cell_summary_rows = []
    for (area, cell_idx), g in pred_df.groupby(['area', 'cell_idx']):
        y = g['actual_rsp'].to_numpy()
        yhat = g['pred_rsp'].to_numpy()
        ss_tot = np.sum((y - y.mean()) ** 2)
        cell_summary_rows.append({
            'area': area,
            'cell_idx': int(cell_idx),
            'r2': np.nan if ss_tot <= 0 else float(
                1 - np.sum((y - yhat) ** 2) / ss_tot
            ),
        })
    return pd.DataFrame(cell_summary_rows)


def load_raw_pred_df():
    pred_pkl = fit_savepath / 'normalization_bubble_raw_pred.pkl'
    if pred_pkl.exists():
        return pd.read_pickle(pred_pkl)
    return None


def format_param_text(row, cell_r2_full, pred_row=None):
    lines = [
        f"area={row['area']}  cell_idx={int(row['cell_idx'])}  "
        f"global_idx={int(row['global_idx'])}",
        f"site={row['site_name']}  local={int(row['local_cell_idx'])}",
        f"object={int(row['object_id'])}  success={row['success']}",
        '',
        f"R2_bubble={row['r2']:.3f}",
        f"R2_full_pred={cell_r2_full:.3f}" if np.isfinite(cell_r2_full) else 'R2_full_pred=nan',
        '',
        f"b={row['b']:.3f}  k={row['k']:.3f}  sigma={row['sigma']:.3f}",
        f"active_x={row['active_x']:.1f}  active_y={row['active_y']:.1f}  "
        f"active_std={row['active_std']:.1f}",
        f"neg_x={row['negative_x']:.1f}  neg_y={row['negative_y']:.1f}  "
        f"neg_std={row['negative_std']:.1f}",
    ]
    if pred_row is not None:
        lines.extend([
            '',
            f"actual_full={pred_row['actual_rsp']:.3f}  "
            f"pred_full={pred_row['pred_rsp']:.3f}",
        ])
    return '\n'.join(lines)


def plot_one_cell_image(
    row,
    cell_meta,
    img,
    cell_r2_full,
    pred_row,
    out_path,
):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    active_map = gaussian_map(
        xx, yy, row['active_x'], row['active_y'], row['active_std']
    )
    neg_map = gaussian_map(
        xx, yy, row['negative_x'], row['negative_y'], row['negative_std']
    )
    active_vis = active_map / active_map.max()
    neg_vis = neg_map / neg_map.max()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.imshow(
        active_vis, cmap='Reds', vmin=0, vmax=1,
        alpha=OVERLAY_ALPHA * active_vis,
    )
    ax.imshow(
        neg_vis, cmap='Blues', vmin=0, vmax=1,
        alpha=OVERLAY_ALPHA * neg_vis,
    )

    title_r2_full = (
        f'{cell_r2_full:.3f}' if np.isfinite(cell_r2_full) else 'nan'
    )
    ax.set_title(
        f"{row['area']} global={int(cell_meta['global_idx'])} "
        f"obj={int(row['object_id'])}  "
        f"R2_bubble={row['r2']:.3f}  R2_full={title_r2_full}",
        fontsize=11,
    )
    ax.text(
        0.02, 0.98,
        format_param_text(row, cell_r2_full, pred_row),
        transform=ax.transAxes,
        va='top', ha='left',
        color='white', fontsize=8, family='monospace',
        bbox=dict(facecolor='black', alpha=0.55, edgecolor='none'),
    )
    ax.axis('off')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


#%%
fit_df = pd.read_pickle(fit_savepath / 'normalization_bubble_fit.pkl')
pred_df = load_raw_pred_df()
pred_cell_df = load_raw_pred_cell_df()

if pred_cell_df is None:
    print(
        'Warning: raw prediction files not found. '
        'R2_full_pred will be nan. Run Cell 10 in Normalization_Model_Bubble.py first.'
    )
    pred_cell_df = pd.DataFrame(columns=['area', 'cell_idx', 'r2'])

cell_r2_lookup = {
    (r['area'], int(r['cell_idx'])): float(r['r2'])
    for _, r in pred_cell_df.iterrows()
}

pred_lookup = {}
if pred_df is not None:
    for _, r in pred_df.iterrows():
        key = (r['area'], int(r['cell_idx']), int(r['img_id']))
        pred_lookup[key] = r

object_ids = sorted(fit_df['object_id'].unique().astype(int).tolist())
if MAX_OBJECTS is not None:
    object_ids = object_ids[:MAX_OBJECTS]

fig_savepath.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SAMPLE_SEED)
n_saved = 0
n_skipped = 0
sample_records = []

for area in AREAS:
    area_df = fit_df.loc[fit_df['area'] == area].copy()
    all_cell_ids = sorted(area_df['cell_idx'].unique().astype(int).tolist())
    cell_ids = sample_cell_ids(all_cell_ids, N_CELLS_PER_AREA, rng)

    for cell_idx in cell_ids:
        meta = area_df.loc[area_df['cell_idx'] == cell_idx].iloc[0]
        sample_records.append({
            'area': area,
            'cell_idx': cell_idx,
            'global_idx': int(meta['global_idx']),
            'site_name': meta['site_name'],
        })

    print(
        f'Plotting {area}: {len(cell_ids)} / {len(all_cell_ids)} cells '
        f'x {len(object_ids)} objects'
    )

    for cell_idx in tqdm(cell_ids, desc=f'{area} cells'):
        cell_rows = area_df.loc[area_df['cell_idx'] == cell_idx]
        if cell_rows.empty:
            continue

        cell_meta = cell_rows.iloc[0]
        global_idx = int(cell_meta['global_idx'])
        cell_r2_full = cell_r2_lookup.get((area, cell_idx), np.nan)

        for obj_id in object_ids:
            row_match = cell_rows.loc[cell_rows['object_id'] == obj_id]
            if row_match.empty:
                continue
            row = row_match.iloc[0]

            out_path = (
                fig_savepath / area
                / f'cell_{global_idx:04d}'
                / f'obj_{obj_id:02d}.png'
            )
            if SKIP_EXISTING and out_path.exists():
                n_skipped += 1
                continue

            img_file = raw_img_path / f'{obj_id:04d}.jpg'
            if not img_file.exists():
                raise FileNotFoundError(f'Raw image not found: {img_file}')

            img = plt.imread(img_file)
            pred_row = pred_lookup.get((area, cell_idx, obj_id))

            plot_one_cell_image(
                row=row,
                cell_meta=cell_meta,
                img=img,
                cell_r2_full=cell_r2_full,
                pred_row=pred_row,
                out_path=out_path,
            )
            n_saved += 1

if sample_records:
    sample_df = pd.DataFrame(sample_records)
    sample_path = fig_savepath / 'sampled_cells.csv'
    sample_df.to_csv(sample_path, index=False, encoding='utf-8-sig')
    print(f'Sampled cell list saved to {sample_path}')

print(f'Done. saved={n_saved}, skipped={n_skipped}, output={fig_savepath}')

#%%



