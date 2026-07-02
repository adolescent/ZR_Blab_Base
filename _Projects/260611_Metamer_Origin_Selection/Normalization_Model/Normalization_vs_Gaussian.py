'''
Normalization vs Gaussian model fitting with 5-fold cross-validation.

For each neuron × object, fit both models on three data subsets:
  bubble_rest, bubble_only, rest_only

Each fit reports in-sample r2 (full data) and cv_r2 (mean test R2 across 5 folds).
'''
#%%
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


savepath = Path(
    r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Normalization_Bubble'
)
datapath = Path(
    r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Metamer_SingleBubble'
)
mask_file = Path(
    r'C:\#working_folder\#Codes\ZR_Blab_Base\Py_Structure\Info_Files\Masks_Metamer_Singlebubble_v251107.npz'
)

AREAS = ['ML', 'MSB', 'AL', 'ASB']
DATA_SUBSETS = ['bubble_rest', 'bubble_only', 'rest_only']

MASK_DOWNSAMPLE = 8
N_STARTS = 4
MAX_NFEV = 500
MIN_RSP_VAR = 1e-8
N_CV_FOLDS = 5
CV_SEED = 42

MAX_CELLS_PER_AREA = None
MAX_OBJECTS = None

FIT_RSP_SOURCE = 'avr_rsp'

savepath.mkdir(parents=True, exist_ok=True)

#%%
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

    print(f'Loaded layout: objects={len(object_ids)}')
    print(f'Loaded masks: {masks.shape}, visible fraction={masks.mean():.4f}')
    return layout, masks, object_index


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


layout, masks_raw, object_index = load_shared_layout_and_masks()
object_masks = prepare_object_masks(masks_raw, object_index, factor=MASK_DOWNSAMPLE)
fit_grid_h, fit_grid_w = next(iter(object_masks.values())).shape[1:]
raw_grid_h, raw_grid_w = masks_raw.shape[1:]

print(
    f'Prepared object masks: {len(object_masks)} objects, '
    f'fit grid={fit_grid_h}x{fit_grid_w}, downsample={MASK_DOWNSAMPLE}'
)

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


def calc_fit_metrics(y_true, y_pred):
    residual = y_pred - y_true
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    return r2, rmse, mae


def normalization_predict(theta, mask_stack):
    b, k, sigma, active_x, active_y, active_std, neg_x, neg_y, neg_std = theta
    mask_flat = mask_stack.reshape(mask_stack.shape[0], -1)
    active_kernel = gaussian_kernel_2d(active_x, active_y, active_std)
    negative_kernel = gaussian_kernel_2d(neg_x, neg_y, neg_std)
    active_drive = mask_flat @ active_kernel
    negative_drive = mask_flat @ negative_kernel
    return b + k * active_drive / (sigma + negative_drive)


def gaussian_sum_predict(theta, mask_stack):
    b, k, active_x, active_y, active_std = theta
    mask_flat = mask_stack.reshape(mask_stack.shape[0], -1)
    active_kernel = gaussian_kernel_2d(active_x, active_y, active_std)
    active_drive = mask_flat @ active_kernel
    return b + k * active_drive


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


def _run_least_squares_fit(y, mask_stack, predict_fn, build_starts, bounds_fn):
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

    lower, upper = bounds_fn(y)
    best = None
    best_cost = np.inf

    def residual_func(theta):
        return predict_fn(theta, mask_stack) - y

    for x0 in build_starts(y, mask_stack):
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

    y_pred = predict_fn(best.x, mask_stack)
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


def _norm_build_starts(y, mask_stack):
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


def _norm_bounds(y):
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = max(y_max - y_min, 1.0)
    lower = [y_min - y_range, 0.0, 1e-4, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
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


def _gauss_build_starts(y, mask_stack):
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = max(y_max - y_min, 1.0)
    active_center = response_weighted_center(mask_stack, y, mode='high')
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


def _gauss_bounds(y):
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


def fit_one_normalization_model(y, mask_stack):
    return _run_least_squares_fit(
        y, mask_stack, normalization_predict, _norm_build_starts, _norm_bounds
    )


def fit_one_gaussian_sum_model(y, mask_stack):
    return _run_least_squares_fit(
        y, mask_stack, gaussian_sum_predict, _gauss_build_starts, _gauss_bounds
    )


def cross_val_r2(y, mask_stack, fit_fn, predict_fn, n_splits=N_CV_FOLDS, seed=CV_SEED):
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    y = y[finite]
    mask_stack = mask_stack[finite]

    if len(y) < n_splits:
        return np.nan

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, n_splits)
    fold_r2s = []

    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        theta, _, _ = fit_fn(y[train_idx], mask_stack[train_idx])
        if theta is None:
            fold_r2s.append(np.nan)
            continue
        y_pred = predict_fn(theta, mask_stack[test_idx])
        r2, _, _ = calc_fit_metrics(y[test_idx], y_pred)
        fold_r2s.append(r2)

    return float(np.nanmean(fold_r2s))


NORM_PARAM_NAMES = [
    'b', 'k', 'sigma',
    'active_x_fit', 'active_y_fit', 'active_std_fit',
    'negative_x_fit', 'negative_y_fit', 'negative_std_fit',
]
GAUSS_PARAM_NAMES = ['b', 'k', 'active_x_fit', 'active_y_fit', 'active_std_fit']


def params_to_record(theta, model_name):
    record = {
        'b': np.nan, 'k': np.nan, 'sigma': np.nan,
        'active_x_fit': np.nan, 'active_y_fit': np.nan, 'active_std_fit': np.nan,
        'negative_x_fit': np.nan, 'negative_y_fit': np.nan, 'negative_std_fit': np.nan,
        'active_x': np.nan, 'active_y': np.nan, 'active_std': np.nan,
        'negative_x': np.nan, 'negative_y': np.nan, 'negative_std': np.nan,
    }
    if theta is None:
        return record

    if model_name == 'normalization':
        for name, value in zip(NORM_PARAM_NAMES, theta):
            record[name] = float(value)
        record['active_x'] = float((theta[3] + 0.5) * MASK_DOWNSAMPLE - 0.5)
        record['active_y'] = float((theta[4] + 0.5) * MASK_DOWNSAMPLE - 0.5)
        record['active_std'] = float(theta[5] * MASK_DOWNSAMPLE)
        record['negative_x'] = float((theta[6] + 0.5) * MASK_DOWNSAMPLE - 0.5)
        record['negative_y'] = float((theta[7] + 0.5) * MASK_DOWNSAMPLE - 0.5)
        record['negative_std'] = float(theta[8] * MASK_DOWNSAMPLE)
    elif model_name == 'gaussian_sum':
        for name, value in zip(GAUSS_PARAM_NAMES, theta):
            record[name] = float(value)
        record['active_x'] = float((theta[2] + 0.5) * MASK_DOWNSAMPLE - 0.5)
        record['active_y'] = float((theta[3] + 0.5) * MASK_DOWNSAMPLE - 0.5)
        record['active_std'] = float(theta[4] * MASK_DOWNSAMPLE)
    return record


MODELS = {
    'normalization': (fit_one_normalization_model, normalization_predict),
    'gaussian_sum': (fit_one_gaussian_sum_model, gaussian_sum_predict),
}


def subset_n_counts(subset, idx):
    if subset == 'bubble_rest':
        return int(idx['n_bubble']), int(idx['n_rest'])
    if subset == 'bubble_only':
        return int(idx['n_bubble']), 0
    return 0, int(idx['n_rest'])


def fit_all():
    records = []

    for area in AREAS:
        avr_rsp, cell_info = load_area_data(area)
        print(f'Fitting {area}: {avr_rsp.shape[0]} cells')

        for cell_idx in tqdm(range(avr_rsp.shape[0]), desc=f'{area} cells'):
            cell_meta = cell_info.iloc[cell_idx].to_dict()

            for object_id, idx in object_index.items():
                for subset in DATA_SUBSETS:
                    cols, mask_stack = get_subset_data(
                        subset, idx, object_masks[object_id]
                    )
                    y = avr_rsp[cell_idx, cols]
                    n_bubble, n_rest = subset_n_counts(subset, idx)

                    for model_name, (fit_fn, pred_fn) in MODELS.items():
                        theta, _, fit_info = fit_fn(y, mask_stack)
                        cv_r2 = cross_val_r2(y, mask_stack, fit_fn, pred_fn)

                        record = {
                            'area': area,
                            'cell_idx': int(cell_idx),
                            'object_id': int(object_id),
                            'img_id': int(object_id),
                            'data_subset': subset,
                            'model_name': model_name,
                            'n_bubble': n_bubble,
                            'n_rest': n_rest,
                            'rsp_mean': float(np.nanmean(y)),
                            'rsp_std': float(np.nanstd(y)),
                            'mask_downsample': int(MASK_DOWNSAMPLE),
                            'fit_grid_h': int(fit_grid_h),
                            'fit_grid_w': int(fit_grid_w),
                            'raw_grid_h': int(raw_grid_h),
                            'raw_grid_w': int(raw_grid_w),
                            'fit_rsp_source': FIT_RSP_SOURCE,
                            'cv_r2': cv_r2,
                        }
                        record.update(cell_meta)
                        record.update(fit_info)
                        record.update(params_to_record(theta, model_name))
                        records.append(record)

    return pd.DataFrame(records)


fit_df = fit_all()

out_pkl = savepath / 'normalization_vs_gaussian_cv_fit.pkl'
out_csv = savepath / 'normalization_vs_gaussian_cv_fit.csv'
fit_df.to_pickle(out_pkl)
fit_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

print(f'Saved {len(fit_df)} rows to {out_pkl}')
for model_name in MODELS:
    for subset in DATA_SUBSETS:
        sub = fit_df[
            (fit_df['model_name'] == model_name) & (fit_df['data_subset'] == subset)
        ]
        print(
            f'  {model_name} [{subset}]: n={len(sub)}, '
            f'median r2={sub["r2"].median(skipna=True):.3f}, '
            f'median cv_r2={sub["cv_r2"].median(skipna=True):.3f}'
        )

fit_df.head()

#%%

