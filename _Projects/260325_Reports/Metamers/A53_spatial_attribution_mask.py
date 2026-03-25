'''
Spatial attribution masks (Wang & Ponce, 2026, NN; Fig. 3c–d).
At each AlexNet conv5 location (i, j), regress neural responses across images
onto F[:, :, i, j] (256-D); map adjusted R² over 13×13.

Backbone: conv5_unpooled (1000, 256, 13, 13) — see Alex_Response_conv5_Notpool.py.
'''

#%%
import OS_Tools as ot
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- defaults (override in build_bundle) ---
DATAFOLDER = r'E:\#Preprocessed_Data\Selected_Cells'
CONV5_NPZ = 'Alex_conv5_unpooled.npz'
CONV5_KEY = 'conv5_unpooled'
SAVE_NAME = 'spatial_attribution_Fig3.npz'


def default_metamer_conditions_df():
    img_indices = np.tile(np.arange(1, 41), 25)
    shuffle_levels = np.tile(np.repeat(np.arange(5), 40), 5)
    return pd.DataFrame({'Img_Index': img_indices, 'Shuffle_Level': shuffle_levels})


def default_raw_ani_ids(df=None, shuffle_levels=0, img_min=1, img_max=20):
    """
    Select trial ids by shuffle level(s) and image-index range.

    Defaults reproduce previous behavior:
        shuffle_levels=0, img_min=1, img_max=20  -> 100 trials.

    Args:
        df: condition dataframe with columns ['Img_Index', 'Shuffle_Level'].
        shuffle_levels: int or iterable of ints, e.g. 0 or [0, 1, 2].
        img_min: inclusive lower bound on Img_Index (1..40).
        img_max: inclusive upper bound on Img_Index (1..40).
    """
    if df is None:
        df = default_metamer_conditions_df()

    if np.isscalar(shuffle_levels):
        shuf = [int(shuffle_levels)]
    else:
        shuf = [int(x) for x in shuffle_levels]
    if len(shuf) == 0:
        raise ValueError('shuffle_levels cannot be empty.')
    if img_min > img_max:
        raise ValueError('img_min must be <= img_max.')

    mask = (
        df['Shuffle_Level'].isin(shuf)
        & (df['Img_Index'] >= int(img_min))
        & (df['Img_Index'] <= int(img_max))
    )
    return df.index[mask].to_numpy()


def effective_df_ridge(X, alpha):
    """
    Effective df for ridge on standardized X (no intercept column): 1 + sum_j s_j^2/(s_j^2+alpha).
    Intercept adds ~1; sklearn Ridge centers y and X before solving.
    """
    n, p = X.shape
    if n < 2:
        return np.nan
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return 1.0 + float(np.sum(s**2 / (s**2 + alpha)))


def adjusted_r2_ridge(y, y_hat, X_scaled, alpha):
    """Generalized adj R² using effective df of ridge hat matrix on X_scaled."""
    y = np.asarray(y, dtype=float).ravel()
    y_hat = np.asarray(y_hat, dtype=float).ravel()
    n = y.size
    if n < 3 or np.var(y) < 1e-12:
        return np.nan
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - y_hat) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    df_eff = effective_df_ridge(X_scaled, alpha)
    if df_eff >= n - 1:
        return np.nan
    return float(1.0 - (ss_res / (n - df_eff)) / (ss_tot / (n - 1)))


def scaler_mean_std_for_locations(back_rsp, fit_indices):
    """StandardScaler stats per (i,j) on rows fit_indices — shape (13, 13, 256) each."""
    fit_indices = np.asarray(fit_indices, dtype=int)
    mean = np.zeros((13, 13, 256), dtype=float)
    scale = np.ones((13, 13, 256), dtype=float)
    for i in range(13):
        for j in range(13):
            X = back_rsp[fit_indices, :, i, j]
            m = X.mean(axis=0)
            s = X.std(axis=0)
            s = np.where(s < 1e-12, 1.0, s)
            mean[i, j] = m
            scale[i, j] = s
    return mean, scale


def apply_scaler_row(x_256, mean_256, scale_256):
    return (x_256 - mean_256) / scale_256


def fit_cell_spatial_maps(
    y_trials,
    back_rsp,
    fit_indices,
    ridge_alpha=1,
):
    """
    y_trials: (1000,) or subset passed as full length aligned to axis 0 of back_rsp
    back_rsp: (1000, 256, 13, 13)
    fit_indices: (n_fit,) indices into dim 0
    Returns:
        adj_r2_map (13, 13), r2_map (13, 13), coefs (13, 13, 256), intercepts (13, 13)
    """
    y_full = np.asarray(y_trials, dtype=float).ravel()
    assert back_rsp.shape[0] == y_full.size
    fit_indices = np.asarray(fit_indices, dtype=int)
    y = y_full[fit_indices]
    n_fit = fit_indices.size

    adj_r2_map = np.full((13, 13), np.nan, dtype=float)
    r2_map = np.full((13, 13), np.nan, dtype=float)
    coefs = np.zeros((13, 13, 256), dtype=float)
    intercepts = np.zeros((13, 13), dtype=float)

    if np.var(y) < 1e-12:
        return adj_r2_map, r2_map, coefs, intercepts

    for i in range(13):
        for j in range(13):
            X = back_rsp[fit_indices, :, i, j]
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
            ridge.fit(Xs, y)
            y_hat = ridge.predict(Xs)
            r2_map[i, j] = ridge.score(Xs, y)
            adj_r2_map[i, j] = adjusted_r2_ridge(y, y_hat, Xs, ridge_alpha)
            coefs[i, j, :] = ridge.coef_
            intercepts[i, j] = ridge.intercept_

    return adj_r2_map, r2_map, coefs, intercepts


def load_metamer_responses(datafolder, brain_sites, bubble_suffix='_Cells_Bubble'):
    """metamer_rsp[site] shape (N_cell, 1000), normalized per cell."""
    metamer_rsp = {}
    for site in brain_sites:
        psth = np.load(ot.Join(datafolder, f'{site}{bubble_suffix}.npz'), allow_pickle=True)['psth']
        r = psth[:, :1000, 160:320].sum(-1).astype(float)
        r = r / np.maximum(r.max(1, keepdims=True), 1e-12)
        metamer_rsp[site] = r
    return metamer_rsp


def load_conv5_backbone(datafolder, npz_name=CONV5_NPZ, key=CONV5_KEY):
    path = ot.Join(datafolder, npz_name)
    back_rsp = np.load(path, allow_pickle=True)[key]
    assert back_rsp.shape == (1000, 256, 13, 13), back_rsp.shape
    return back_rsp


def build_spatial_attribution_bundle(
    datafolder=DATAFOLDER,
    brain_sites=None,
    fit_indices=None,
    ridge_alpha=1,
    metamer_suffix='_Cells_Bubble',
    conv5_npz=CONV5_NPZ,
    save_path=None,
    save_coefs=True,
):
    """
    Compute adj R² maps for all cells × sites. Saves .npz with arrays and metadata.
    """
    if brain_sites is None:
        brain_sites = ['AL', 'ASB', 'ML', 'MSB']
    if fit_indices is None:
        fit_indices = default_raw_ani_ids()

    fit_indices = np.asarray(fit_indices, dtype=int)
    back_rsp = load_conv5_backbone(datafolder, conv5_npz, CONV5_KEY)
    metamer_rsp = load_metamer_responses(datafolder, brain_sites, metamer_suffix)
    scaler_mean, scaler_std = scaler_mean_std_for_locations(back_rsp, fit_indices)

    bundle = {
        'fit_indices': fit_indices,
        'ridge_alpha': float(ridge_alpha),
        'brain_sites': np.array(brain_sites, dtype=object),
        'conv5_npz': conv5_npz,
        'version': 'Fig3_spatial_attribution_v1',
        'scaler_mean': scaler_mean,
        'scaler_std': scaler_std,
    }

    for site in brain_sites:
        R = metamer_rsp[site]
        assert R.shape[1] == 1000, R.shape
        n_cells = R.shape[0]
        adj_maps = np.zeros((n_cells, 13, 13), dtype=float)
        r2_maps = np.zeros((n_cells, 13, 13), dtype=float)
        coefs_all = np.zeros((n_cells, 13, 13, 256), dtype=float) if save_coefs else None
        int_all = np.zeros((n_cells, 13, 13), dtype=float) if save_coefs else None

        for c in tqdm(range(n_cells), desc=f'{site} cells'):
            adj, r2, coef, inter = fit_cell_spatial_maps(
                R[c], back_rsp, fit_indices, ridge_alpha=ridge_alpha
            )
            adj_maps[c] = adj
            r2_maps[c] = r2
            if save_coefs:
                coefs_all[c] = coef
                int_all[c] = inter

        bundle[f'adj_r2_{site}'] = adj_maps
        bundle[f'r2_{site}'] = r2_maps
        if save_coefs:
            bundle[f'coefs_{site}'] = coefs_all
            bundle[f'intercepts_{site}'] = int_all

    if save_path is None:
        save_path = ot.Join(datafolder, SAVE_NAME)
    np.savez_compressed(save_path, **bundle)
    print(f'Saved bundle: {save_path}')
    return bundle, save_path


class SpatialAttributionStore:
    """
    Query Fig. 3 adj R² maps via get_mask. For local_contribution_map(img_id), attach_backbone.
    predict_map_new_image uses saved scaler + coefs only (no backbone).
    """

    def __init__(self, bundle_dict):
        self._d = bundle_dict
        self._sites = list(np.atleast_1d(bundle_dict['brain_sites']))
        self._back_rsp = None

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        return cls(d)

    def attach_backbone(self, back_rsp):
        """(1000, 256, 13, 13) for local_contribution."""
        assert back_rsp.shape == (1000, 256, 13, 13)
        self._back_rsp = back_rsp

    def get_mask(self, site, cell_id):
        """Fig. 3 style: (13, 13) adjusted R² map (constant across img_id)."""
        key = f'adj_r2_{site}'
        if key not in self._d:
            raise KeyError(key)
        return np.asarray(self._d[key][cell_id])

    def get_r2_mask(self, site, cell_id):
        key = f'r2_{site}'
        return np.asarray(self._d[key][cell_id])

    def _mean_std_maps(self):
        return np.asarray(self._d['scaler_mean']), np.asarray(self._d['scaler_std'])

    def local_contribution_map(self, site, cell_id, img_id):
        """
        Per-location linear projection for one image (not adj R²): w_ij^T x_ij + b_ij.
        Uses saved scaler_mean/std (same as StandardScaler on fit_indices). Needs coefs in bundle.
        """
        if f'coefs_{site}' not in self._d:
            raise RuntimeError('Bundle was saved with save_coefs=False; re-run build with save_coefs=True.')
        coefs = np.asarray(self._d[f'coefs_{site}'][cell_id])
        inter = np.asarray(self._d[f'intercepts_{site}'][cell_id])
        mean, std = self._mean_std_maps()
        if self._back_rsp is None:
            raise RuntimeError('Call attach_backbone(back_rsp) first.')
        X_img = self._back_rsp[int(img_id), :, :, :]
        out = np.zeros((13, 13), dtype=float)
        for i in range(13):
            for j in range(13):
                xs = apply_scaler_row(X_img[:, i, j], mean[i, j], std[i, j])
                out[i, j] = float(np.dot(coefs[i, j], xs) + inter[i, j])
        return out

    def predict_map_new_image(self, site, cell_id, conv5_hw):
        """
        conv5_hw: (256, 13, 13) activations for one image (same preprocessing as backbone script).
        Uses scaler_mean/std stored in bundle (no full backbone required).
        """
        if f'coefs_{site}' not in self._d:
            raise RuntimeError('Bundle was saved with save_coefs=False; re-run build with save_coefs=True.')
        coefs = np.asarray(self._d[f'coefs_{site}'][cell_id])
        inter = np.asarray(self._d[f'intercepts_{site}'][cell_id])
        mean, std = self._mean_std_maps()
        out = np.zeros((13, 13), dtype=float)
        for i in range(13):
            for j in range(13):
                xs = apply_scaler_row(conv5_hw[:, i, j], mean[i, j], std[i, j])
                out[i, j] = float(np.dot(coefs[i, j], xs) + inter[i, j])
        return out


def extract_alexnet_conv5_unpooled(image_path, device=None):
    """Single-image conv5 (256, 13, 13); same as Alex_Response_conv5_Notpool."""
    import torch
    from torchvision import models, transforms
    from torchvision.models.feature_extraction import create_feature_extractor
    from PIL import Image as PILImage

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = tfm(PILImage.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    alexnet.eval()
    fetcher = create_feature_extractor(alexnet, return_nodes={'features.11': 'conv5_unpooled'})
    with torch.no_grad():
        z = fetcher(img)['conv5_unpooled'].cpu().numpy()[0]
    return z


def predict_new_image_from_path(store, site, cell_id, image_path):
    """Convenience: extract conv5 then predict_map_new_image."""
    conv5 = extract_alexnet_conv5_unpooled(image_path)
    return store.predict_map_new_image(site, cell_id, conv5)


def contribution_summary_for_images(store, site, cell_id, img_ids):
    """
    Build per-image contribution maps + concentration scores.
    Requires store.attach_backbone(...) beforehand.
    Returns:
        maps: dict[int, (13,13)]
        summary_df: columns [img_id, abs_sum, peak_abs, top10_ratio]
    """
    maps = {}
    rows = []
    for img_id in img_ids:
        m = store.local_contribution_map(site, cell_id, int(img_id))
        maps[int(img_id)] = m
        a = np.abs(m).ravel()
        top10_thresh = np.quantile(a, 0.90)
        top10_ratio = a[a >= top10_thresh].sum() / (a.sum() + 1e-12)
        rows.append(
            {
                'img_id': int(img_id),
                'abs_sum': float(a.sum()),
                'peak_abs': float(a.max()),
                'top10_ratio': float(top10_ratio),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values('img_id').reset_index(drop=True)
    return maps, summary_df


def plot_contribution_maps(maps_dict, summary_df=None, cmap='vlag', center=0.0, ncols=4,vmax=None,vmin=None):
    """Plot multiple image-specific contribution maps with seaborn heatmap."""
    img_ids = sorted(maps_dict.keys())
    n = len(img_ids)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    sns.set_theme(style='white', context='notebook')
    for k, img_id in enumerate(img_ids):
        ax = axes[k]
        m = maps_dict[img_id]
        title = f'img {img_id}'
        if summary_df is not None and len(summary_df) > 0:
            row = summary_df.loc[summary_df['img_id'] == img_id]
            if len(row) == 1:
                title += f"\ntop10={float(row['top10_ratio'].iloc[0]):.3f}"
        sns.heatmap(
            m,
            ax=ax,
            cmap=cmap,
            center=center,
            vmax=vmax,
            vmin=vmin,
            square=True,
            cbar=(k == 0),
            cbar_kws={'label': 'local projection'} if k == 0 else None,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel('conv5 j')
        ax.set_ylabel('conv5 i')
    for k in range(n, len(axes)):
        axes[k].axis('off')
    plt.tight_layout()
    return fig


#%% main: build bundle (uncomment to run)
if __name__ == '__main__':
    bundle, path = build_spatial_attribution_bundle(
        datafolder=DATAFOLDER,brain_sites=['ASB'],fit_indices=default_raw_ani_ids(shuffle_levels=[0,1,2,3,4],img_min=1,img_max=40),
        ridge_alpha=1000,
        save_path=ot.Join(r'E:\#Preprocessed_Data\260305_Report_Data\Bubbles', SAVE_NAME),
    )
#%% show example
    bundle_path = r'E:\#Preprocessed_Data\260305_Report_Data\Bubbles\spatial_attribution_Fig3.npz'
    store = SpatialAttributionStore.load(bundle_path)
    site = 'ASB'
    cell_id = 210
    fig3_mask = store.get_mask(site, cell_id)

    sns.set_theme(style="white", context="notebook")
    plt.figure(figsize=(5, 4.2))
    sns.heatmap(
        fig3_mask,
        cmap="rocket",
        square=True,
        cbar_kws={"label": "Adj. R²"},
        linewidths=0,
    )
    plt.title("Adj. R² mask (Fig. 3 style)")
    plt.xlabel("conv5 j")
    plt.ylabel("conv5 i")
    plt.tight_layout()
    plt.show()
#%%
    # Example: compare per-image contribution maps for one neuron
    back_rsp = load_conv5_backbone(DATAFOLDER)
    store.attach_backbone(back_rsp)
    img_ids = [2]
    maps, summary = contribution_summary_for_images(store, site, cell_id, img_ids)
    print(summary)
    plot_contribution_maps(maps, summary_df=summary, cmap='vlag', ncols=3)
    plt.show()

#%% Then I want to mask each cell's 
all_img_path = ot.Get_File_Name(r'E:\#Stimsets\Metamer_P4_C4321_Object_STI150_1300','.jpg')[:1000]

