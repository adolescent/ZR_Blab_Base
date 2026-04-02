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
from PIL import Image, ImageFilter

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
    conv5_npz=CONV5_NPZ,conv5_key=CONV5_KEY,
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
    back_rsp = load_conv5_backbone(datafolder, conv5_npz, conv5_key)
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
    ## CORE BUILD PART
    DATAFOLDER = r'E:\#Preprocessed_Data\Selected_Cells'
    CONV5_NPZ = 'Alex_Response_conv5_unpooled_nsd.npz'
    CONV5_KEY = 'conv5_unpooled'
    SAVE_NAME = 'spatial_attribution_Fig3_NSD_1k.npz'
    bundle, path = build_spatial_attribution_bundle(
        datafolder=DATAFOLDER,brain_sites=['MSB'],metamer_suffix='_NSD_Demo_Part',
        # fit_indices=default_raw_ani_ids(shuffle_levels=[0,1,2,3,4],img_min=1,img_max=40),
        fit_indices=np.arange(1000),
        conv5_npz=CONV5_NPZ,conv5_key=CONV5_KEY,
        ridge_alpha=1000,
        save_path=ot.Join(r'E:\#Preprocessed_Data\260305_Report_Data\Bubbles', SAVE_NAME),
    )
#%% show example
    bundle_path = r'E:\#Preprocessed_Data\260305_Report_Data\Bubbles\spatial_attribution_Fig3_NSD_1k.npz'
    store = SpatialAttributionStore.load(bundle_path)
    site = 'MSB'
    cell_id = 73
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
    # plt.title("Adj. R² mask (Fig. 3 style)")
    # plt.xlabel("conv5 j")
    # plt.ylabel("conv5 i")
    plt.tight_layout()
    plt.show()
#%% ### Plot contribution map
    # Example: compare per-image contribution maps for one neuron
    back_rsp = load_conv5_backbone(DATAFOLDER)
    store.attach_backbone(back_rsp)
    img_ids = [2]
    maps, summary = contribution_summary_for_images(store, site, cell_id, img_ids)
    print(summary)
    plot_contribution_maps(maps, summary_df=summary, cmap='vlag', ncols=3)
    plt.show()

#%% Single cell masked images.
    # all_img_path = ot.Get_File_Name(r'E:\#Stimsets\Metamer_P4_C4321_Object_STI150_1300','.jpg')[:1000]
    all_img_path = ot.Get_File_Name(r'E:\#Stimsets\NSD1000','.bmp')[:1000]
    # top-k masked images by cell response
    def binary_mask_top_percent(mask_13x13, keep_ratio=0.2):
        """
        Binarize adj-R2 mask and keep top keep_ratio pixels (default 20%).
        Returns uint8 mask in {0,1} shape (13,13).
        """
        m = np.asarray(mask_13x13, dtype=float)
        if not (0 < keep_ratio <= 1):
            raise ValueError('keep_ratio must be in (0,1].')
        thr = np.quantile(m, 1 - keep_ratio)
        return (m >= thr).astype(np.uint8)


    def keep_connected_components(bin_mask_13x13, min_component_size=3, connectivity=4, keep_mode='all'):
        """
        Remove small connected components from a binary mask.

        Args:
            bin_mask_13x13: uint8/boolean mask in {0,1}, shape (13,13)
            min_component_size: minimum number of pixels in a component to keep
            connectivity: 4 or 8
            keep_mode: 'all' -> keep every component with size>=min_component_size
                    'largest' -> keep only the largest component (after filtering)
        """
        m = (np.asarray(bin_mask_13x13) > 0).astype(np.uint8)
        if m.ndim != 2:
            raise ValueError('bin_mask_13x13 must be 2D.')
        H, W = m.shape
        if connectivity == 4:
            nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif connectivity == 8:
            nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            raise ValueError("connectivity must be 4 or 8.")

        visited = np.zeros((H, W), dtype=bool)
        out = np.zeros((H, W), dtype=np.uint8)
        best_coords = None
        best_size = -1

        for y in range(H):
            for x in range(W):
                if m[y, x] == 0 or visited[y, x]:
                    continue
                # BFS for one component
                q = [(y, x)]
                visited[y, x] = True
                coords = [(y, x)]
                qi = 0
                while qi < len(q):
                    cy, cx = q[qi]
                    qi += 1
                    for dy, dx in nbrs:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and (not visited[ny, nx]) and m[ny, nx] == 1:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                            coords.append((ny, nx))

                comp_size = len(coords)
                if comp_size < int(min_component_size):
                    continue
                if keep_mode == 'all':
                    for (yy, xx) in coords:
                        out[yy, xx] = 1
                elif keep_mode == 'largest':
                    if comp_size > best_size:
                        best_size = comp_size
                        best_coords = coords
                else:
                    raise ValueError("keep_mode must be 'all' or 'largest'.")

        if keep_mode == 'largest' and best_coords is not None:
            for (yy, xx) in best_coords:
                out[yy, xx] = 1

        return out


    def topk_img_ids_for_cell_response(metamer_rsp_site, cell_id, k=20, descending=True):
        """
        metamer_rsp_site: (N_cell, 1000)
        Return top-k img ids ranked by this cell response.
        """
        rsp = np.asarray(metamer_rsp_site[cell_id], dtype=float).ravel()
        order = np.argsort(rsp)
        if descending:
            order = order[::-1]
        k = int(min(max(1, k), rsp.size))
        return order[:k], rsp[order[:k]]


    def resize_mask_to_image(mask_13x13, out_hw, mode='bicubic', blur_radius=1.2):
        """
        Resize 13x13 mask to (H,W) with interpolation and optional blur.

        mode: 'nearest' | 'bilinear' | 'bicubic'
        blur_radius: Gaussian blur radius after resize (0 disables blur).
        """
        h, w = int(out_hw[0]), int(out_hw[1])
        m = np.clip(mask_13x13.astype(np.float32), 0.0, 1.0)
        pil = Image.fromarray((m * 255).astype(np.uint8))
        resample_map = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
        }
        if mode not in resample_map:
            raise ValueError("mode must be one of ['nearest', 'bilinear', 'bicubic']")
        pil = pil.resize((w, h), resample=resample_map[mode])
        if blur_radius is not None and float(blur_radius) > 0:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        m = np.asarray(pil, dtype=np.float32) / 255.0
        return m


    def masked_topk_images_for_cell(
        store,
        metamer_rsp_site,
        site,
        cell_id,
        all_img_path,
        k=20,
        keep_ratio=0.2,
        out_size=(224, 224),
        min_component_size=3,
        keep_mode='all',
        smooth_mode='bicubic',
        smooth_blur_radius=1.2,
    ):
        """
        Build masked images for top-k activating images of one cell.
        Returns:
            out: dict with keys:
                img_ids, responses, binary_mask_13x13, masked_imgs_uint8 (k,H,W,3), orig_imgs_uint8 (k,H,W,3)
        """
        fig3_mask = store.get_mask(site, cell_id)
        bin13 = binary_mask_top_percent(fig3_mask, keep_ratio=keep_ratio)
        bin13 = keep_connected_components(
            bin13,
            min_component_size=min_component_size,
            connectivity=4,
            keep_mode=keep_mode,
        )
        top_ids, top_rsp = topk_img_ids_for_cell_response(metamer_rsp_site, cell_id, k=k, descending=True)

        orig_list, masked_list = [], []
        for img_id in top_ids:
            img = Image.open(all_img_path[int(img_id)]).convert('RGB')
            if out_size is not None:
                img = img.resize((out_size[1], out_size[0]), resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
            mask_hw = resize_mask_to_image(
                bin13,
                arr.shape[:2],
                mode=smooth_mode,
                blur_radius=smooth_blur_radius,
            )[..., None]
            masked = (arr.astype(np.float32) * mask_hw).clip(0, 255).astype(np.uint8)
            orig_list.append(arr)
            masked_list.append(masked)

        return {
            'img_ids': np.asarray(top_ids, dtype=int),
            'responses': np.asarray(top_rsp, dtype=float),
            'binary_mask_13x13': bin13,
            'orig_imgs_uint8': np.stack(orig_list, axis=0),
            'masked_imgs_uint8': np.stack(masked_list, axis=0),
        }


    def visualize_mask_and_topk(mask_pack, max_show=12):
        """Visualize binary mask and top-k masked images."""
        img_ids = mask_pack['img_ids']
        rsps = mask_pack['responses']
        orig = mask_pack['orig_imgs_uint8']
        masked = mask_pack['masked_imgs_uint8']
        bin13 = mask_pack['binary_mask_13x13']

        sns.set_theme(style='white')
        plt.figure(figsize=(4.2, 3.8))
        sns.heatmap(bin13, cmap='gray_r', square=True, cbar=False, linewidths=0)
        plt.title('Binary mask (top 20% adj-R2)')
        plt.xlabel('conv5 j')
        plt.ylabel('conv5 i')
        plt.tight_layout()
        plt.show()

        n = int(min(len(img_ids), max_show))
        fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 4.8))
        if n == 1:
            axes = np.asarray(axes).reshape(2, 1)
        for i in range(n):
            axes[0, i].imshow(orig[i])
            axes[0, i].set_title(f'img={img_ids[i]}\nrsp={rsps[i]:.3f}')
            axes[0, i].axis('off')
            axes[1, i].imshow(masked[i])
            axes[1, i].set_title('masked')
            axes[1, i].axis('off')
        fig.suptitle('Top-k activating images (original vs masked)', y=1.02)
        plt.tight_layout()
        plt.show()


    def save_masked_matrix(mask_pack, save_path):
        """
        Save masked-image matrix and metadata.
        masked_imgs_uint8 shape: (k, H, W, 3)
        """
        np.savez_compressed(
            save_path,
            img_ids=mask_pack['img_ids'],
            responses=mask_pack['responses'],
            binary_mask_13x13=mask_pack['binary_mask_13x13'],
            masked_imgs_uint8=mask_pack['masked_imgs_uint8'],
            orig_imgs_uint8=mask_pack['orig_imgs_uint8'],
        )
        print(f'Saved masked matrix: {save_path}')

    #%%
    # Example usage:
    cell_id = 73
    metamer_rsp = load_metamer_responses(DATAFOLDER, [site])
    pack = masked_topk_images_for_cell(
        store,
        metamer_rsp_site=metamer_rsp[site],
        site=site,
        cell_id=cell_id,
        all_img_path=all_img_path,
        k=7,              # adjustable
        keep_ratio=0.2,    # top 20% binary mask
        out_size=(224,224),
        smooth_mode='bilinear',
        smooth_blur_radius=3,min_component_size=3,
    )
    visualize_mask_and_topk(pack, max_show=20)
    # save_masked_matrix(pack, ot.Join(r'E:\#Preprocessed_Data\260305_Report_Data\Bubbles', f'{site}_cell{cell_id}_topk_masked.npz'))


