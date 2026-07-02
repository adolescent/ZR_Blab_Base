'''
Simple Gaussian sum model for bubble/rest responses.

Rsp = b + k * (Ec · Active_kernel)

5 parameters: b, k, active_x, active_y, active_std.
Fit per neuron × per image (20 objects), with three data subsets:
  bubble_rest, bubble_only, rest_only
'''

#%% Cell 1 - imports and paths
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Normalization_Bubble\Simple_Gaussian_Model'
datapath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_SingleBubble'
mask_file = r'C:\#working_folder\#Codes\ZR_Blab_Base\Py_Structure\Info_Files\Masks_Metamer_Singlebubble_v251107.npz'
raw_img_path = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Stimsets\Raw_Objs')

#%% Cell 2 - fitting config
AREAS = ['ML', 'MSB', 'AL', 'ASB']

RUN_MODE = 'batch'           # 'test' | 'batch'
TEST_CELLS_PER_AREA = 5
DATA_SUBSETS = ['bubble_rest', 'bubble_only', 'rest_only']

MASK_DOWNSAMPLE = 8
N_STARTS = 4
MAX_NFEV = 500
MIN_RSP_VAR = 1e-8
MAX_OBJECTS = None

MODEL_NAME = 'gaussian_sum'
FIT_RSP_SOURCE = 'avr_rsp'

MAX_CELLS_PER_AREA = TEST_CELLS_PER_AREA if RUN_MODE == 'test' else None

savepath = Path(savepath)
datapath = Path(datapath)
mask_file = Path(mask_file)
savepath.mkdir(parents=True, exist_ok=True)


#%% Cell 3 - load layout, masks, and responses
def _as_slice(slice_arr):
    values = np.asarray(slice_arr, dtype=int).ravel().tolist()
    if len(values) == 2:
        start, stop = values
        step = None
    elif len(values) == 3:
        start, stop, step = values
    else:
        raise ValueError(f'Cannot convert layout slice entry with values={values}.')
    return slice(start, stop, step)


def load_shared_layout_and_masks():
    layout = np.load(datapath / 'stim_layout.npz', allow_pickle=True)
    masks = np.load(mask_file)['masks'].astype(np.float32)

    if masks.ndim != 3:
        raise ValueError(f'Expected masks with shape (n_img, h, w), got {masks.shape}.')
    if masks.shape[0] < 4540:
        raise ValueError(f'Expected at least 4540 masks, got {masks.shape[0]}.')

    bubble_sl = _as_slice(layout['slice_bubble'])
    rest_sl = _as_slice(layout['slice_rest'])
    data_ids = np.asarray(layout['data_ids'], dtype=int)
    object_id = np.asarray(layout['object_id'])
    stim_type = np.asarray(layout['stim_type']).astype(str)

    object_ids = sorted([
        int(obj) for obj in pd.unique(object_id[bubble_sl])
        if pd.notna(obj)
    ])
    if MAX_OBJECTS is not None:
        object_ids = object_ids[:MAX_OBJECTS]

    object_index = {}
    for obj in object_ids:
        bubble_cols = np.where((object_id == obj) & (stim_type == 'bubble'))[0]
        rest_cols = np.where((object_id == obj) & (stim_type == 'rest'))[0]

        if len(bubble_cols) != 80 or len(rest_cols) != 80:
            warnings.warn(
                f'Object {obj}: expected 80 bubble and 80 rest samples, '
                f'got {len(bubble_cols)} and {len(rest_cols)}.'
            )

        object_index[obj] = {
            'bubble_cols': bubble_cols,
            'rest_cols': rest_cols,
            'cols': np.concatenate([bubble_cols, rest_cols]),
            'raw_mask_ids': data_ids[np.concatenate([bubble_cols, rest_cols])],
            'n_bubble': len(bubble_cols),
            'n_rest': len(rest_cols),
        }

    validate_bubble_grouping(layout, masks, object_index)

    print(f'Loaded layout: bubble={bubble_sl}, rest={rest_sl}, objects={len(object_ids)}')
    print(f'Loaded masks: {masks.shape}, visible fraction={masks.mean():.4f}')
    return layout, masks, object_index


def validate_bubble_grouping(layout, masks, object_index):
    data_ids = np.asarray(layout['data_ids'], dtype=int)
    stim_type = np.asarray(layout['stim_type']).astype(str)

    expected_data_ids = np.arange(300, 4540)
    if not np.array_equal(data_ids, expected_data_ids):
        raise ValueError('layout data_ids are not the expected raw stim indices 300:4540.')

    expected_blocks = {
        'metamer': (0, 1000, 300, 1299),
        'metamer_tail': (1000, 1040, 1300, 1339),
        'bubble': (1040, 2640, 1340, 2939),
        'rest': (2640, 4240, 2940, 4539),
    }
    for block_name, (start, stop, raw_start, raw_stop) in expected_blocks.items():
        block_types = stim_type[start:stop]
        if not np.all(block_types == block_name):
            raise ValueError(f'{block_name} block has unexpected stim_type values.')
        if int(data_ids[start]) != raw_start or int(data_ids[stop - 1]) != raw_stop:
            raise ValueError(f'{block_name} block raw mask ids are not aligned.')

    if len(object_index) != 20:
        raise ValueError(f'Expected 20 bubble/rest objects, got {len(object_index)}.')

    for obj, idx in object_index.items():
        bubble_cols = idx['bubble_cols']
        rest_cols = idx['rest_cols']
        if len(bubble_cols) != 80 or len(rest_cols) != 80:
            raise ValueError(
                f'Object {obj}: expected 80 bubble and 80 rest samples, '
                f'got {len(bubble_cols)} and {len(rest_cols)}.'
            )
        if not np.all(np.diff(bubble_cols) == 1) or not np.all(np.diff(rest_cols) == 1):
            raise ValueError(f'Object {obj}: bubble/rest samples are not contiguous.')
        if not np.allclose(masks[data_ids[bubble_cols]], 1.0 - masks[data_ids[rest_cols]]):
            raise ValueError(f'Object {obj}: bubble and rest masks are not paired complements.')

    print('Grouping check passed: 300 FOB + 1000 metamer + 40 tail + 1600 bubble + 1600 rest.')
    print('Mask pairing check passed: each object has 80 bubble/rest pairs and paired masks are complements.')


def load_area_data(area):
    area_path = datapath / area
    avr_rsp = np.load(area_path / 'avr_rsp.npy').astype(np.float64)
    cell_info = pd.read_csv(area_path / 'cell_site_info.csv')

    if len(cell_info) != avr_rsp.shape[0]:
        raise ValueError(
            f'{area}: cell_site_info rows ({len(cell_info)}) do not match '
            f'avr_rsp cells ({avr_rsp.shape[0]}).'
        )

    return avr_rsp, cell_info


layout, masks_raw, object_index = load_shared_layout_and_masks()


#%% Cell 4 - prepare object masks
def downsample_masks(mask_stack, factor=MASK_DOWNSAMPLE):
    if factor == 1:
        return mask_stack.astype(np.float64)

    n_img, h, w = mask_stack.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f'Mask size {(h, w)} is not divisible by factor={factor}.')

    new_h = h // factor
    new_w = w // factor
    return (
        mask_stack.reshape(n_img, new_h, factor, new_w, factor)
        .mean(axis=(2, 4))
        .astype(np.float64)
    )


def prepare_object_masks(masks, object_index, factor=MASK_DOWNSAMPLE):
    prepared = {}
    for obj, idx in object_index.items():
        prepared[obj] = downsample_masks(masks[idx['raw_mask_ids']], factor=factor)
    return prepared


object_masks = prepare_object_masks(masks_raw, object_index, factor=MASK_DOWNSAMPLE)
fit_grid_h, fit_grid_w = next(iter(object_masks.values())).shape[1:]
raw_grid_h, raw_grid_w = masks_raw.shape[1:]

print(
    f'Prepared object masks: {len(object_masks)} objects, '
    f'fit grid={fit_grid_h}x{fit_grid_w}, downsample={MASK_DOWNSAMPLE}'
)


def get_subset_data(subset, idx, mask_stack):
    if subset == 'bubble_rest':
        cols = idx['cols']
        masks = mask_stack
    elif subset == 'bubble_only':
        cols = idx['bubble_cols']
        masks = mask_stack[:idx['n_bubble']]
    elif subset == 'rest_only':
        cols = idx['rest_cols']
        masks = mask_stack[idx['n_bubble']:]
    else:
        raise ValueError(f'Unknown subset: {subset}')
    return cols, masks


def subset_save_dir(subset):
    if RUN_MODE == 'test':
        return savepath / 'test' / subset
    return savepath / subset


#%% Cell 5 - model and fit functions
Y_GRID, X_GRID = np.mgrid[0:fit_grid_h, 0:fit_grid_w]
X_FLAT = X_GRID.ravel().astype(np.float64)
Y_FLAT = Y_GRID.ravel().astype(np.float64)


def gaussian_kernel_2d(x0, y0, std):
    std = max(float(std), 1e-6)
    dist2 = (X_FLAT - x0) ** 2 + (Y_FLAT - y0) ** 2
    kernel = np.exp(-0.5 * dist2 / (std ** 2))
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        return np.full_like(kernel, 1.0 / kernel.size)
    return kernel / kernel_sum


def gaussian_sum_predict(theta, mask_stack):
    b, k, active_x, active_y, active_std = theta
    mask_flat = mask_stack.reshape(mask_stack.shape[0], -1)
    active_kernel = gaussian_kernel_2d(active_x, active_y, active_std)
    active_drive = mask_flat @ active_kernel
    return b + k * active_drive


def calc_fit_metrics(y_true, y_pred):
    residual = y_pred - y_true
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    return r2, rmse, mae


def response_weighted_center(mask_stack, response):
    mask_flat = mask_stack.reshape(mask_stack.shape[0], -1)
    response = np.asarray(response, dtype=np.float64)
    weights = response - np.nanmin(response)
    weights = np.clip(weights, 0, None)

    spatial_weight = weights @ mask_flat
    total = spatial_weight.sum()
    if total <= 0:
        return (fit_grid_w - 1) / 2, (fit_grid_h - 1) / 2

    x0 = float((spatial_weight @ X_FLAT) / total)
    y0 = float((spatial_weight @ Y_FLAT) / total)
    return x0, y0


def build_initial_points(y, mask_stack):
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = max(y_max - y_min, 1.0)
    active_center = response_weighted_center(mask_stack, y)
    center = ((fit_grid_w - 1) / 2, (fit_grid_h - 1) / 2)
    std_small = max(min(fit_grid_h, fit_grid_w) / 8, 1.0)
    std_mid = max(min(fit_grid_h, fit_grid_w) / 4, 1.0)
    std_large = max(min(fit_grid_h, fit_grid_w) / 2.5, 1.0)

    starts = [
        [y_min, y_range, active_center[0], active_center[1], std_mid],
        [y_min, y_range, center[0], center[1], std_mid],
        [y_min, y_range, active_center[0], active_center[1], std_small],
        [float(np.nanmean(y)), y_range, center[0], center[1], std_large],
    ]
    return starts[:N_STARTS]


def parameter_bounds(y):
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = max(y_max - y_min, 1.0)
    lower = [y_min - y_range, 0.0, 0.0, 0.0, 1.0]
    upper = [
        y_max + y_range,
        max(y_max + y_range, 10 * y_range),
        fit_grid_w - 1,
        fit_grid_h - 1,
        max(fit_grid_h, fit_grid_w),
    ]
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def fit_one_gaussian_sum_model(y, mask_stack):
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    y = y[finite]
    mask_stack = mask_stack[finite]

    if len(y) < 20:
        return None, None, {
            'success': False,
            'message': 'too_few_valid_observations',
            'n_obs': int(len(y)),
        }

    if float(np.nanvar(y)) < MIN_RSP_VAR:
        y_pred = np.full_like(y, float(np.nanmean(y)))
        r2, rmse, mae = calc_fit_metrics(y, y_pred)
        return None, y_pred, {
            'success': False,
            'message': 'low_response_variance',
            'n_obs': int(len(y)),
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cost': np.nan,
            'nfev': 0,
        }

    lower, upper = parameter_bounds(y)
    best = None
    best_cost = np.inf

    def residual_func(theta):
        return gaussian_sum_predict(theta, mask_stack) - y

    for x0 in build_initial_points(y, mask_stack):
        x0 = np.clip(np.asarray(x0, dtype=np.float64), lower, upper)
        try:
            result = least_squares(
                residual_func,
                x0=x0,
                bounds=(lower, upper),
                max_nfev=MAX_NFEV,
                loss='soft_l1',
            )
        except Exception as exc:
            warnings.warn(f'least_squares failed from one start: {exc}')
            continue

        if result.cost < best_cost:
            best = result
            best_cost = result.cost

    if best is None:
        return None, None, {
            'success': False,
            'message': 'all_starts_failed',
            'n_obs': int(len(y)),
        }

    y_pred = gaussian_sum_predict(best.x, mask_stack)
    r2, rmse, mae = calc_fit_metrics(y, y_pred)
    fit_info = {
        'success': bool(best.success),
        'message': str(best.message),
        'n_obs': int(len(y)),
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cost': float(best.cost),
        'nfev': int(best.nfev),
    }
    return best.x, y_pred, fit_info


#%% Cell 6 - area fitting loop helpers
PARAM_NAMES = ['b', 'k', 'active_x_fit', 'active_y_fit', 'active_std_fit']


def params_to_record(theta):
    record = {name: np.nan for name in PARAM_NAMES}
    record.update({'active_x': np.nan, 'active_y': np.nan, 'active_std': np.nan})

    if theta is None:
        return record

    for name, value in zip(PARAM_NAMES, theta):
        record[name] = float(value)

    record['active_x'] = float((theta[2] + 0.5) * MASK_DOWNSAMPLE - 0.5)
    record['active_y'] = float((theta[3] + 0.5) * MASK_DOWNSAMPLE - 0.5)
    record['active_std'] = float(theta[4] * MASK_DOWNSAMPLE)
    return record


def sample_test_cells(avr_rsp, rng):
    n_cells = avr_rsp.shape[0]
    n_test = min(TEST_CELLS_PER_AREA, n_cells)
    return rng.choice(n_cells, size=n_test, replace=False)


def fit_area(area, subset, save_dir):
    avr_rsp, cell_info = load_area_data(area)
    records = []

    if RUN_MODE == 'test':
        rng = np.random.default_rng(42)
        cell_indices = sample_test_cells(avr_rsp, rng)
        print(f'Test fitting {area} [{subset}]: sampled cells = {cell_indices.tolist()}')
    else:
        cell_indices = np.arange(avr_rsp.shape[0])

    for cell_idx in tqdm(cell_indices, desc=f'{area} {subset}'):
        cell_idx = int(cell_idx)
        cell_meta = cell_info.iloc[cell_idx].to_dict()

        for object_id, idx in object_index.items():
            cols, mask_stack = get_subset_data(subset, idx, object_masks[object_id])
            y = avr_rsp[cell_idx, cols]
            theta, _, fit_info = fit_one_gaussian_sum_model(y, mask_stack)

            record = {
                'area': area,
                'cell_idx': cell_idx,
                'object_id': int(object_id),
                'img_id': int(object_id),
                'data_subset': subset,
                'n_bubble': int(idx['n_bubble']) if subset != 'rest_only' else 0,
                'n_rest': int(idx['n_rest']) if subset != 'bubble_only' else 0,
                'rsp_mean': float(np.nanmean(y)),
                'rsp_std': float(np.nanstd(y)),
                'mask_downsample': int(MASK_DOWNSAMPLE),
                'fit_grid_h': int(fit_grid_h),
                'fit_grid_w': int(fit_grid_w),
                'raw_grid_h': int(raw_grid_h),
                'raw_grid_w': int(raw_grid_w),
                'fit_rsp_source': FIT_RSP_SOURCE,
                'model_name': MODEL_NAME,
                'run_mode': RUN_MODE,
                'is_test_fit': RUN_MODE == 'test',
            }
            record.update(cell_meta)
            record.update(fit_info)
            record.update(params_to_record(theta))
            records.append(record)

    area_df = pd.DataFrame(records)
    area_df.to_pickle(save_dir / f'gaussian_sum_fit_{area}.pkl')
    area_df.to_csv(save_dir / f'gaussian_sum_fit_{area}.csv', index=False, encoding='utf-8-sig')
    return area_df


#%% Cell 7 - run all fits and save DataFrame
all_results = {}

for subset in DATA_SUBSETS:
    subset_dir = subset_save_dir(subset)
    subset_dir.mkdir(parents=True, exist_ok=True)

    subset_dfs = []
    for area in AREAS:
        print(f'Fitting {area} [{subset}]...')
        area_df = fit_area(area, subset, subset_dir)
        subset_dfs.append(area_df)
        print(
            f'{area}: {len(area_df)} fits, '
            f'success={area_df["success"].mean():.3f}, '
            f'median R2={area_df["r2"].median(skipna=True):.3f}'
        )

    fit_df = pd.concat(subset_dfs, ignore_index=True)
    fit_df.to_pickle(subset_dir / 'gaussian_sum_fit.pkl')
    fit_df.to_csv(subset_dir / 'gaussian_sum_fit.csv', index=False, encoding='utf-8-sig')
    all_results[subset] = fit_df
    print(f'Saved {len(fit_df)} fitted models to {subset_dir}')

print(f'All subsets done. RUN_MODE={RUN_MODE}')


#%% Cell 8 - visualize fitted active field on raw image
SHOW_AREA = 'MSB'
SHOW_CELL = 14   
SHOW_OBJECT = 9
SHOW_SUBSET = 'bubble_rest'
USE_TEST_FIT = True
OVERLAY_ALPHA = 0.7

vis_dir = subset_save_dir(SHOW_SUBSET) if USE_TEST_FIT else savepath / SHOW_SUBSET
fit_df_vis = pd.read_pickle(vis_dir / 'gaussian_sum_fit.pkl')

area_cells = (
    fit_df_vis.loc[fit_df_vis['area'] == SHOW_AREA]
    .drop_duplicates('cell_idx')
    .sort_values('cell_idx')
    .reset_index(drop=True)
)
print('Available cells in this area (SHOW_CELL uses iloc index):')
print(area_cells[['cell_idx', 'global_idx', 'local_cell_idx', 'site_name', 'data_subset']])

if SHOW_CELL < 0 or SHOW_CELL >= len(area_cells):
    raise IndexError(
        f'SHOW_CELL={SHOW_CELL} is out of range for {SHOW_AREA}; '
        f'valid range is 0..{len(area_cells) - 1}.'
    )

cell_meta = area_cells.iloc[SHOW_CELL]
actual_cell_idx = int(cell_meta['cell_idx'])
real_cell_id = int(cell_meta['global_idx'])

row = fit_df_vis.query(
    'area == @SHOW_AREA and cell_idx == @actual_cell_idx and object_id == @SHOW_OBJECT'
)
if row.empty:
    raise ValueError(
        f'No fit found for area={SHOW_AREA}, list cell={SHOW_CELL} '
        f'(global_idx={real_cell_id}), object_id={SHOW_OBJECT}, subset={SHOW_SUBSET}.'
    )
row = row.iloc[0]

img = plt.imread(raw_img_path / f'{SHOW_OBJECT:04d}.jpg')
h, w = img.shape[:2]
yy, xx = np.mgrid[0:h, 0:w]


def gaussian_map(x0, y0, std):
    std = max(float(std), 1e-6)
    g = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / std ** 2)
    return g / g.sum()


active_map = gaussian_map(row['active_x'], row['active_y'], row['active_std'])
active_map /= active_map.max()

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img)
ax.imshow(active_map, cmap='Reds', vmin=0, vmax=1, alpha=OVERLAY_ALPHA * active_map)
ax.set_title(
    f'{SHOW_AREA} cell_id={real_cell_id} object={SHOW_OBJECT} subset={SHOW_SUBSET}  '
    f'R2={row["r2"]:.3f}  success={row["success"]}'
)
ax.text(
    0.02, 0.98,
    f'cell_id={real_cell_id}\nsite={cell_meta["site_name"]}\nlocal={int(cell_meta["local_cell_idx"])}',
    transform=ax.transAxes,
    va='top', ha='left',
    color='white', fontsize=10,
    bbox=dict(facecolor='black', alpha=0.45, edgecolor='none'),
)
ax.axis('off')
plt.show()

row[
    [
        'area', 'cell_idx', 'global_idx', 'local_cell_idx', 'site_name',
        'object_id', 'data_subset', 'r2', 'success',
        'b', 'k', 'active_x', 'active_y', 'active_std',
    ]
]

#%%
