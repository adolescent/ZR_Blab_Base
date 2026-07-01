'''
使用normalization model 的方法，和bubble的数据，来计算神经元响应是否满足normalization model，即存在激活和抑制野。

这部分是全新的分析，不应该从前面的文件中寻找分析方法。

normalization 公式：

Rsp=b+k (Ec*Active_kernel)/(sigma+Ec*Negative_kernel)
其中，Ec是刺激的能量，Active_kernel是激活野，Negative_kernel是抑制野，sigma,b和k是模型参数。
Active和negative kernel都使用一个2D 单位gaussian函数进行拟合，假设xy方向的std一样，则只需要三个参数：中心的xy坐标，std。

因此，我们需要拟合的参数有：
1. sigma,b和k
2. Active_kernel的中心的xy坐标，std
3. Negative_kernel的中心的xy坐标，std
共9个参数，使用非线性最小二乘法进行拟合。

Ec是当前mask ON的区域，激活野和抑制野在mask内的响应的和作为输入。




'''


#%% Cell 1 - imports and paths
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Normalization_Bubble'

datapath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_SingleBubble'

mask_file = r'C:\#working_folder\#Codes\ZR_Blab_Base\Py_Structure\Info_Files\Masks_Metamer_Singlebubble_v251107.npz'

#%% Cell 2 - fitting config
# Fitting config. Set MAX_* to a small number for a smoke test, then switch back
# to None for the full run.
AREAS = ['ML', 'MSB', 'AL', 'ASB']

MASK_DOWNSAMPLE = 8
N_STARTS = 4
MAX_NFEV = 500
MIN_RSP_VAR = 1e-8

MAX_CELLS_PER_AREA = None
MAX_OBJECTS = None

MODEL_NAME = 'divisive_normalization_bubble'
FIT_RSP_SOURCE = 'avr_rsp'

savepath = Path(savepath)
datapath = Path(datapath)
mask_file = Path(mask_file)
savepath.mkdir(parents=True, exist_ok=True)


#%% Cell 3 - load layout, masks, and responses
def _as_slice(slice_arr):
    """Convert a layout npz [start, stop] or [start, stop, step] entry to a slice."""
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

    if MAX_CELLS_PER_AREA is not None:
        avr_rsp = avr_rsp[:MAX_CELLS_PER_AREA]
        cell_info = cell_info.iloc[:MAX_CELLS_PER_AREA].copy()

    return avr_rsp, cell_info


layout, masks_raw, object_index = load_shared_layout_and_masks()


#%% Cell 4 - prepare object masks
def downsample_masks(mask_stack, factor=MASK_DOWNSAMPLE):
    """Block-average masks. Values remain in [0, 1] as visible-pixel energy."""
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


def normalization_predict(theta, mask_stack):
    b, k, sigma, active_x, active_y, active_std, neg_x, neg_y, neg_std = theta
    mask_flat = mask_stack.reshape(mask_stack.shape[0], -1)

    active_kernel = gaussian_kernel_2d(active_x, active_y, active_std)
    negative_kernel = gaussian_kernel_2d(neg_x, neg_y, neg_std)

    active_drive = mask_flat @ active_kernel
    negative_drive = mask_flat @ negative_kernel
    return b + k * active_drive / (sigma + negative_drive)


def calc_fit_metrics(y_true, y_pred):
    residual = y_pred - y_true
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    return r2, rmse, mae


def response_weighted_center(mask_stack, response, mode='high'):
    mask_flat = mask_stack.reshape(mask_stack.shape[0], -1)
    response = np.asarray(response, dtype=np.float64)

    if mode == 'high':
        weights = response - np.nanmin(response)
    else:
        weights = np.nanmax(response) - response
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
    active_center = response_weighted_center(mask_stack, y, mode='high')
    neg_center = response_weighted_center(mask_stack, y, mode='low')

    center = ((fit_grid_w - 1) / 2, (fit_grid_h - 1) / 2)
    std_small = max(min(fit_grid_h, fit_grid_w) / 8, 1.0)
    std_mid = max(min(fit_grid_h, fit_grid_w) / 4, 1.0)
    std_large = max(min(fit_grid_h, fit_grid_w) / 2.5, 1.0)

    starts = [
        [y_min, y_range, 0.10, active_center[0], active_center[1], std_mid, neg_center[0], neg_center[1], std_mid],
        [y_min, y_range, 0.25, center[0], center[1], std_mid, center[0], center[1], std_large],
        [y_min, y_range, 0.05, active_center[0], active_center[1], std_small, center[0], center[1], std_large],
        [float(np.nanmean(y)), y_range, 0.50, center[0], center[1], std_large, neg_center[0], neg_center[1], std_small],
    ]
    return starts[:N_STARTS]


def parameter_bounds(y):
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = max(y_max - y_min, 1.0)
    lower = [
        y_min - y_range,
        0.0,
        1e-4,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    ]
    upper = [
        y_max + y_range,
        max(y_max + y_range, 10 * y_range),
        10.0,
        fit_grid_w - 1,
        fit_grid_h - 1,
        max(fit_grid_h, fit_grid_w),
        fit_grid_w - 1,
        fit_grid_h - 1,
        max(fit_grid_h, fit_grid_w),
    ]
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def fit_one_normalization_model(y, mask_stack):
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
        return normalization_predict(theta, mask_stack) - y

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

    y_pred = normalization_predict(best.x, mask_stack)
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
PARAM_NAMES = [
    'b',
    'k',
    'sigma',
    'active_x_fit',
    'active_y_fit',
    'active_std_fit',
    'negative_x_fit',
    'negative_y_fit',
    'negative_std_fit',
]


def params_to_record(theta):
    record = {name: np.nan for name in PARAM_NAMES}
    record.update({
        'active_x': np.nan,
        'active_y': np.nan,
        'active_std': np.nan,
        'negative_x': np.nan,
        'negative_y': np.nan,
        'negative_std': np.nan,
    })

    if theta is None:
        return record

    for name, value in zip(PARAM_NAMES, theta):
        record[name] = float(value)

    # Convert from downsampled fitting-grid coordinates back to raw mask pixels.
    record['active_x'] = float((theta[3] + 0.5) * MASK_DOWNSAMPLE - 0.5)
    record['active_y'] = float((theta[4] + 0.5) * MASK_DOWNSAMPLE - 0.5)
    record['active_std'] = float(theta[5] * MASK_DOWNSAMPLE)
    record['negative_x'] = float((theta[6] + 0.5) * MASK_DOWNSAMPLE - 0.5)
    record['negative_y'] = float((theta[7] + 0.5) * MASK_DOWNSAMPLE - 0.5)
    record['negative_std'] = float(theta[8] * MASK_DOWNSAMPLE)
    return record


def fit_area(area):
    avr_rsp, cell_info = load_area_data(area)
    records = []

    area_iter = tqdm(range(avr_rsp.shape[0]), desc=f'{area} cells')
    for cell_idx in area_iter:
        cell_meta = cell_info.iloc[cell_idx].to_dict()

        for object_id, idx in object_index.items():
            y = avr_rsp[cell_idx, idx['cols']]
            theta, _, fit_info = fit_one_normalization_model(y, object_masks[object_id])

            record = {
                'area': area,
                'cell_idx': int(cell_idx),
                'object_id': int(object_id),
                'img_id': int(object_id),
                'n_bubble': int(idx['n_bubble']),
                'n_rest': int(idx['n_rest']),
                'rsp_mean': float(np.nanmean(y)),
                'rsp_std': float(np.nanstd(y)),
                'mask_downsample': int(MASK_DOWNSAMPLE),
                'fit_grid_h': int(fit_grid_h),
                'fit_grid_w': int(fit_grid_w),
                'raw_grid_h': int(raw_grid_h),
                'raw_grid_w': int(raw_grid_w),
                'fit_rsp_source': FIT_RSP_SOURCE,
                'model_name': MODEL_NAME,
            }
            record.update(cell_meta)
            record.update(fit_info)
            record.update(params_to_record(theta))
            records.append(record)

    area_df = pd.DataFrame(records)
    area_df.to_pickle(savepath / f'normalization_bubble_fit_{area}.pkl')
    area_df.to_csv(savepath / f'normalization_bubble_fit_{area}.csv', index=False, encoding='utf-8-sig')
    return area_df


#%% Cell 7 - run all fits and save DataFrame
all_area_dfs = []
for area in AREAS:
    print(f'Fitting {area}...')
    area_df = fit_area(area)
    all_area_dfs.append(area_df)
    print(
        f'{area}: {len(area_df)} fits, '
        f'success={area_df["success"].mean():.3f}, '
        f'median R2={area_df["r2"].median(skipna=True):.3f}'
    )

fit_df = pd.concat(all_area_dfs, ignore_index=True)
fit_df.to_pickle(savepath / 'normalization_bubble_fit.pkl')
fit_df.to_csv(savepath / 'normalization_bubble_fit.csv', index=False, encoding='utf-8-sig')

print(f'Saved {len(fit_df)} fitted models to {savepath}')
fit_df.head()


#%% Cell 8 - quick reload check
# Quick reload check. After running the full fit you can use this cell alone
# to recover every model parameter for a given neuron and image/object.
fit_df_loaded = pd.read_pickle(savepath / 'normalization_bubble_fit.pkl')

example_model = (
    fit_df_loaded
    .query('success == True')
    .sort_values('r2', ascending=False)
    .head(1)
)

example_model[
    [
        'area', 'cell_idx', 'object_id', 'r2',
        'b', 'k', 'sigma',
        'active_x', 'active_y', 'active_std',
        'negative_x', 'negative_y', 'negative_std',
    ]
]


#%% Quick Tester
# TEST_CELLS_PER_AREA = 5
# SAVE_TEST_RESULT = True

# test_records = []
# rng = np.random.default_rng()

# for area in AREAS:
#     avr_rsp, cell_info = load_area_data(area)
#     n_test = min(TEST_CELLS_PER_AREA, avr_rsp.shape[0])
#     sampled_cells = rng.choice(avr_rsp.shape[0], size=n_test, replace=False)
#     print(f'Test fitting {area}: sampled cells = {sampled_cells.tolist()}')

#     for cell_idx in tqdm(sampled_cells, desc=f'{area} test cells'):
#         cell_idx = int(cell_idx)
#         cell_meta = cell_info.iloc[cell_idx].to_dict()

#         for object_id, idx in object_index.items():
#             y = avr_rsp[cell_idx, idx['cols']]
#             theta, _, fit_info = fit_one_normalization_model(y, object_masks[object_id])

#             record = {
#                 'area': area,
#                 'cell_idx': cell_idx,
#                 'object_id': int(object_id),
#                 'img_id': int(object_id),
#                 'n_bubble': int(idx['n_bubble']),
#                 'n_rest': int(idx['n_rest']),
#                 'rsp_mean': float(np.nanmean(y)),
#                 'rsp_std': float(np.nanstd(y)),
#                 'mask_downsample': int(MASK_DOWNSAMPLE),
#                 'fit_grid_h': int(fit_grid_h),
#                 'fit_grid_w': int(fit_grid_w),
#                 'raw_grid_h': int(raw_grid_h),
#                 'raw_grid_w': int(raw_grid_w),
#                 'fit_rsp_source': FIT_RSP_SOURCE,
#                 'model_name': MODEL_NAME,
#                 'is_test_fit': True,
#             }
#             record.update(cell_meta)
#             record.update(fit_info)
#             record.update(params_to_record(theta))
#             test_records.append(record)

# fit_df_test = pd.DataFrame(test_records)

# if SAVE_TEST_RESULT:
#     fit_df_test.to_pickle(savepath / 'normalization_bubble_fit_test.pkl')
#     fit_df_test.to_csv(
#         savepath / 'normalization_bubble_fit_test.csv',
#         index=False,
#         encoding='utf-8-sig',
#     )

# print(
#     f'Test fit done: {len(fit_df_test)} models, '
#     f'success={fit_df_test["success"].mean():.3f}, '
#     f'median R2={fit_df_test["r2"].median(skipna=True):.3f}'
# )

# fit_df_test.head()
#%% Cell 9 - visualize fitted active / negative fields on raw image
import matplotlib.pyplot as plt

raw_img_path = Path(r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Stimsets\Raw_Objs')

# ---- choose what to show ----
SHOW_AREA = 'AL'
SHOW_CELL = 1            # row index in the area cell list below (pandas iloc order)
SHOW_OBJECT = 3          # 1-20 ani; raw images are 0001.jpg - 0040.jpg
USE_TEST_FIT = False      # True -> fit_df_test; False -> full fit_df
OVERLAY_ALPHA = 0.7

if USE_TEST_FIT:
    fit_df_vis = fit_df_test if 'fit_df_test' in globals() else pd.read_pickle(
        savepath / 'normalization_bubble_fit_test.pkl'
    )
else:
    fit_df_vis = pd.read_pickle(savepath / 'normalization_bubble_fit.pkl')

area_cells = (
    fit_df_vis.loc[fit_df_vis['area'] == SHOW_AREA]
    .drop_duplicates('cell_idx')
    .sort_values('cell_idx')
    .reset_index(drop=True)
)
print('Available cells in this area (SHOW_CELL uses iloc index):')
print(area_cells[['cell_idx', 'global_idx', 'local_cell_idx', 'site_name']])

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
        f'(global_idx={real_cell_id}), object_id={SHOW_OBJECT}.'
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
neg_map = gaussian_map(row['negative_x'], row['negative_y'], row['negative_std'])
active_map /= active_map.max()
neg_map /= neg_map.max()

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img)
ax.imshow(active_map, cmap='Reds', vmin=0, vmax=1, alpha=OVERLAY_ALPHA * active_map)
ax.imshow(neg_map, cmap='Blues', vmin=0, vmax=1, alpha=OVERLAY_ALPHA * neg_map)
ax.set_title(
    f'{SHOW_AREA} cell_id={real_cell_id} object={SHOW_OBJECT}  '
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
        'object_id', 'r2', 'success',
        'b', 'k', 'sigma',
        'active_x', 'active_y', 'active_std',
        'negative_x', 'negative_y', 'negative_std',
    ]
]

#%% Cell 10 - predict raw full-image responses with bubble-fitted models
# Evaluate normalization model on unshuffled (shuffle level 0) metamer responses only.
USE_TEST_FIT_FOR_PRED = False   # True -> use fit_df_test; False -> full fit_df
SAVE_PRED_RESULT = True

if USE_TEST_FIT_FOR_PRED:
    fit_df_pred = (
        fit_df_test if 'fit_df_test' in globals()
        else pd.read_pickle(savepath / 'normalization_bubble_fit_test.pkl')
    )
else:
    fit_df_pred = pd.read_pickle(savepath / 'normalization_bubble_fit.pkl')

met_img_index = layout['metamer_img_index'][:1000].astype(int)
met_shuffle_level = layout['metamer_shuffle_level'][:1000].astype(int)


def raw_image_cols(img_id):
    return np.where(
        (met_img_index == img_id) & (met_shuffle_level == 0)
    )[0]


def predict_full_image(b, k, sigma):
    # Full visible image: normalized Gaussian drives both equal 1.
    return float(b + k / (sigma + 1.0))


pred_records = []
for area in AREAS:
    avr_rsp, cell_info = load_area_data(area)
    area_models = fit_df_pred.loc[fit_df_pred['area'] == area]

    for cell_idx in tqdm(range(avr_rsp.shape[0]), desc=f'{area} raw pred'):
        cell_meta = cell_info.iloc[cell_idx].to_dict()
        cell_models = area_models.loc[area_models['cell_idx'] == cell_idx]

        for img_id in range(1, 21):
            model_row = cell_models.loc[cell_models['object_id'] == img_id]
            if model_row.empty:
                continue
            model_row = model_row.iloc[0]

            actual_rsp = float(avr_rsp[cell_idx, raw_image_cols(img_id)].mean())
            pred_rsp = predict_full_image(model_row['b'], model_row['k'], model_row['sigma'])

            pred_records.append({
                'area': area,
                'cell_idx': int(cell_idx),
                'img_id': int(img_id),
                'object_id': int(img_id),
                'actual_rsp': actual_rsp,
                'pred_rsp': pred_rsp,
                'residual': actual_rsp - pred_rsp,
                'abs_error': float(abs(actual_rsp - pred_rsp)),
                'bubble_fit_r2': float(model_row['r2']),
                'bubble_fit_success': bool(model_row['success']),
                **{k: cell_meta[k] for k in [
                    'global_idx', 'site_name', 'local_cell_idx',
                    'dprime_face', 'dprime_body', 'ceiling_index',
                ] if k in cell_meta},
            })

pred_df = pd.DataFrame(pred_records)

# Per-neuron summary across 20 raw images
cell_summary_rows = []
for (area, cell_idx), g in pred_df.groupby(['area', 'cell_idx']):
    y = g['actual_rsp'].to_numpy()
    yhat = g['pred_rsp'].to_numpy()
    ss_tot = np.sum((y - y.mean()) ** 2)
    cell_summary_rows.append({
        'area': area,
        'cell_idx': int(cell_idx),
        'n_img': int(len(g)),
        'global_idx': g['global_idx'].iloc[0] if 'global_idx' in g.columns else np.nan,
        'site_name': g['site_name'].iloc[0] if 'site_name' in g.columns else np.nan,
        'rmse': float(np.sqrt(np.mean((y - yhat) ** 2))),
        'mae': float(np.mean(np.abs(y - yhat))),
        'r2': np.nan if ss_tot <= 0 else float(1 - np.sum((y - yhat) ** 2) / ss_tot),
        'median_bubble_fit_r2': float(g['bubble_fit_r2'].median()),
    })

pred_cell_df = pd.DataFrame(cell_summary_rows)

if SAVE_PRED_RESULT:
    pred_df.to_pickle(savepath / 'normalization_bubble_raw_pred.pkl')
    pred_df.to_csv(savepath / 'normalization_bubble_raw_pred.csv', index=False, encoding='utf-8-sig')
    pred_cell_df.to_pickle(savepath / 'normalization_bubble_raw_pred_cell.pkl')
    pred_cell_df.to_csv(
        savepath / 'normalization_bubble_raw_pred_cell.csv',
        index=False, encoding='utf-8-sig',
    )

print(
    f'Raw-image prediction done: {len(pred_df)} rows, '
    f'median abs_error={pred_df["abs_error"].median():.3f}, '
    f'median cell r2={pred_cell_df["r2"].median(skipna=True):.3f}'
)
pred_df.head()

#%% Cell 11 - evaluate active / negative center consistency across 20 images
K_MIN = 0.5                    # only keep fits with k > K_MIN
MIN_IMAGES_PER_CELL = 3        # need at least this many valid images per neuron
USE_TEST_FIT_FOR_CENTER = False
SAVE_CENTER_RESULT = True

if USE_TEST_FIT_FOR_CENTER:
    fit_df_center = (
        fit_df_test if 'fit_df_test' in globals()
        else pd.read_pickle(savepath / 'normalization_bubble_fit_test.pkl')
    )
else:
    fit_df_center = pd.read_pickle(savepath / 'normalization_bubble_fit.pkl')

good_fits = fit_df_center.loc[fit_df_center['k'] > K_MIN].copy()
print(f'Kept {len(good_fits)} / {len(fit_df_center)} fits with k > {K_MIN}')

center_records = []
for (area, cell_idx), g in good_fits.groupby(['area', 'cell_idx']):
    if len(g) < MIN_IMAGES_PER_CELL:
        continue

    ax = g['active_x'].to_numpy(dtype=float)
    ay = g['active_y'].to_numpy(dtype=float)
    nx = g['negative_x'].to_numpy(dtype=float)
    ny = g['negative_y'].to_numpy(dtype=float)

    active_mean_x, active_mean_y = ax.mean(), ay.mean()
    neg_mean_x, neg_mean_y = nx.mean(), ny.mean()

    active_dist = np.sqrt((ax - active_mean_x) ** 2 + (ay - active_mean_y) ** 2)
    neg_dist = np.sqrt((nx - neg_mean_x) ** 2 + (ny - neg_mean_y) ** 2)
    active_neg_dist = np.sqrt((ax - nx) ** 2 + (ay - ny) ** 2)

    center_records.append({
        'area': area,
        'cell_idx': int(cell_idx),
        'global_idx': g['global_idx'].iloc[0] if 'global_idx' in g.columns else np.nan,
        'site_name': g['site_name'].iloc[0] if 'site_name' in g.columns else np.nan,
        'n_img': int(len(g)),
        'active_mean_x': float(active_mean_x),
        'active_mean_y': float(active_mean_y),
        'negative_mean_x': float(neg_mean_x),
        'negative_mean_y': float(neg_mean_y),
        'active_spread_std': float(active_dist.std()),
        'negative_spread_std': float(neg_dist.std()),
        'active_spread_mean': float(active_dist.mean()),
        'negative_spread_mean': float(neg_dist.mean()),
        'active_spread_max': float(active_dist.max()),
        'negative_spread_max': float(neg_dist.max()),
        'mean_center_dist': float(np.sqrt(
            (active_mean_x - neg_mean_x) ** 2 + (active_mean_y - neg_mean_y) ** 2
        )),
        'mean_active_neg_dist': float(active_neg_dist.mean()),
        'median_k': float(g['k'].median()),
        'median_r2': float(g['r2'].median()),
    })

center_consistency_df = pd.DataFrame(center_records)

if SAVE_CENTER_RESULT:
    center_consistency_df.to_pickle(savepath / 'normalization_bubble_center_consistency.pkl')
    center_consistency_df.to_csv(
        savepath / 'normalization_bubble_center_consistency.csv',
        index=False, encoding='utf-8-sig',
    )

print(
    f'Center consistency: {len(center_consistency_df)} neurons, '
    f'median active spread={center_consistency_df["active_spread_std"].median():.2f} px, '
    f'median negative spread={center_consistency_df["negative_spread_std"].median():.2f} px'
)
center_consistency_df.sort_values('active_spread_std').head()

#%%


