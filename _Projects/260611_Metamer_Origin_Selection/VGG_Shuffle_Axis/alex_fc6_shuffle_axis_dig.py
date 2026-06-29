'''
这个脚本旨在寻找 AlexNet 模型中，是否存在特征性的维度，表征 metamer 打乱的程度，大致思路如下：

1. 首先，需要加载 AlexNet 模型，并提取特征。
2. 做样本空间构建，使用 NSD1k 数据，fc6 特征（fc7 前一层），PCA 降维到 50D，展示解释的 VAR 和 PC1-2 载荷最高/最低的图片
3. 将 metamer 图片嵌入上述样本空间，并得到这些图片在空间中的载荷
4. 可视化，展示不同 level 的 shuffle 图片在这些 PC 中的载荷，以及寻找一个 2D 平面，是否能反映图片的打乱程度
'''


#%% 配置

savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\alexnet_space'
nsd_figpath = r'E:\#Stimsets\NSD1000'
metamer_img_path = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Stimsets\Metamer1k'

FEATURE_LAYER = 'fc6'   # 'last_conv' | 'fc6'
N_DIM = 50          # 保留前 50 个主成分
BATCH = 32
N_EXTREME = 10      # PC 上载荷最高/最低各展示几张图

assert FEATURE_LAYER in ('last_conv', 'fc6'), "FEATURE_LAYER must be 'last_conv' or 'fc6'"
FEATURE_TAG = 'last_conv_pre_relu' if FEATURE_LAYER == 'last_conv' else FEATURE_LAYER
CACHE = savepath + rf'\alexnet_nsd1k_step1_{FEATURE_TAG}.npz'

SAVE_FIGURES = True
SHOW_FIGURES = True


#%% 1. 用 NSD1k + AlexNet 建立 50D 样本空间

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import OS_Tools as ot
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as T
from sklearn.decomposition import PCA

# --- NSD1k 图片列表 ---
img_paths = sorted(Path(nsd_figpath).glob('*.bmp')) + sorted(Path(nsd_figpath).glob('*.jpg'))
img_paths = [str(p) for p in img_paths]
assert len(img_paths) == 1000, f'expected 1000 NSD images, got {len(img_paths)}'

if os.path.isfile(CACHE):
    d = np.load(CACHE, allow_pickle=True)
    if int(d['n_dim']) != N_DIM:
        raise ValueError(f'cache n_dim={int(d["n_dim"])}, expected {N_DIM}; delete {CACHE} and rerun')
    if 'feature_layer' in d.files and str(d['feature_layer']) != FEATURE_LAYER:
        raise ValueError(f'cache feature_layer={d["feature_layer"]!r}, expected {FEATURE_LAYER!r}; delete {CACHE} and rerun')
    if 'feats' in d.files:
        feats = d['feats']
    elif FEATURE_LAYER == 'last_conv' and 'last_conv' in d.files:
        feats = d['last_conv']   # 旧版缓存
    elif FEATURE_LAYER == 'fc6' and 'fc6' in d.files:
        feats = d['fc6']         # 旧版缓存
    else:
        raise KeyError(f'no feature array in {CACHE}')
    if FEATURE_LAYER == 'fc6' and feats.min() >= -1e-6:
        raise ValueError(
            f'{CACHE} looks like post-ReLU fc6 (min={feats.min():.4f}); '
            f'delete it and rerun step1 to rebuild pre-ReLU features'
        )
    cumvar = d['cumvar']
    ev_ratio = d['ev_ratio']
    n_dim = int(d['n_dim'])
    coords = d['coords']
    pc_mean = d['pc_mean']
    pc_components = d['pc_components']
    img_paths = list(d['img_paths'])
    print(f'loaded cache: {CACHE}')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()

    def _extract(batch_tensor):
        if FEATURE_LAYER == 'last_conv':
            # features[10] = conv5；hook 里立刻拷贝，避免后面的 inplace ReLU 改掉 pre-ReLU 输出。
            buf = []

            def _hook(m, inp, out):
                buf.append(out.detach().cpu().clone())

            h = model.features[10].register_forward_hook(_hook)
            model.features(batch_tensor)
            h.remove()
            return buf[0].flatten(1)
        # fc6: classifier[1] 线性层输出 → 4096D
        buf = []

        def _hook(m, inp, out):
            # 必须立刻拷贝：classifier[2] 是 inplace ReLU，会原地改掉 fc6 的输出张量
            buf.append(out.detach().cpu())

        h = model.classifier[1].register_forward_hook(_hook)
        model(batch_tensor)
        h.remove()
        return buf[0]

    sample = _extract(preprocess(Image.open(img_paths[0]).convert('RGB')).unsqueeze(0).to(device))
    feat_dim = int(sample.shape[1])
    feats = np.zeros((len(img_paths), feat_dim), np.float32)

    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), BATCH), desc=f'AlexNet {FEATURE_LAYER}'):
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in img_paths[i:i + BATCH]]
            feats[i:i + BATCH] = _extract(torch.stack(imgs).to(device)).numpy()

    # --- PCA: 保留前 50 个主成分 ---
    pca_full = PCA().fit(feats)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ev_ratio = pca_full.explained_variance_ratio_[:N_DIM]
    n_dim = N_DIM
    pc_mean = pca_full.mean_.astype(np.float32)
    pc_components = pca_full.components_[:N_DIM].astype(np.float32)
    coords = pca_full.transform(feats)[:, :N_DIM]

    ot.Mkdir(savepath, mute=True)
    np.savez(CACHE, feats=feats, feature_layer=FEATURE_LAYER, cumvar=cumvar, ev_ratio=ev_ratio,
             n_dim=n_dim, coords=coords, pc_mean=pc_mean, pc_components=pc_components,
             img_paths=np.array(img_paths, dtype=object))
    print(f'saved: {CACHE}')

#%% --- 1) 50 个 PC 解释的方差 ---
layer_label = 'last conv' if FEATURE_LAYER == 'last_conv' else 'fc6'
print(f'\n=== PCA variance (AlexNet {layer_label}, NSD1k, K={N_DIM}) ===')
print(f'feature dim = {feats.shape[1]}')
print(f'50 PCs explain {cumvar[N_DIM - 1]:.2%} of {layer_label} variance')
print(f'PC1 alone: {ev_ratio[0]:.2%}')
print(f'PC1+PC2:   {cumvar[1]:.2%}')
print('Per-PC variance (first 10):')
for k in range(min(10, N_DIM)):
    print(f'  PC{k + 1:2d}: {ev_ratio[k]:.3%}  (cum {cumvar[k]:.2%})')


def _extreme(scores, k=10):
    order = np.argsort(scores)
    return order[-k:][::-1], order[:k]


def _save_fig(fig, name):
    if SAVE_FIGURES:
        fig_dir = ot.Join(savepath, 'figures')
        ot.Mkdir(fig_dir, mute=True)
        path = ot.Join(fig_dir, f'{name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'saved figure: {path}')
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# --- 2) PC1 / PC2 上载荷最大、最小的 10 张图 ---
for plot_pc in (1, 2):
    pc_idx = plot_pc - 1
    scores = coords[:, pc_idx]
    hi_idx, lo_idx = _extreme(scores, N_EXTREME)

    fig, axes = plt.subplots(2, N_EXTREME, figsize=(1.2 * N_EXTREME, 2.8))
    for col, i in enumerate(hi_idx):
        axes[0, col].imshow(Image.open(img_paths[i]))
        axes[0, col].axis('off')
    for col, i in enumerate(lo_idx):
        axes[1, col].imshow(Image.open(img_paths[i]))
        axes[1, col].axis('off')
    axes[0, 0].set_ylabel(f'PC{plot_pc} hi', fontsize=9)
    axes[1, 0].set_ylabel(f'PC{plot_pc} lo', fontsize=9)
    fig.suptitle(f'AlexNet {layer_label} — NSD1k extremes on PC{plot_pc} ({ev_ratio[pc_idx]:.1%} var)', fontsize=10)
    fig.tight_layout()
    _save_fig(fig, f'step1_{FEATURE_LAYER}_pc{plot_pc}_extremes')

# --- 累积方差曲线 ---
fig, ax = plt.subplots(figsize=(3, 2.5))
ax.plot(np.arange(1, len(cumvar) + 1), cumvar, 'k-', lw=1.5)
ax.axvline(N_DIM, color='C1', ls='--', lw=0.8, label=f'K={N_DIM} ({cumvar[N_DIM - 1]:.1%})')
ax.set_xlim(1, 100)
ax.set_xlabel('N PCs')
ax.set_ylabel('Explained VAR')
ax.legend(fontsize=8)
ax.set_title(f'AlexNet {layer_label} PCA (NSD1k)')
fig.tight_layout()
_save_fig(fig, f'step1_{FEATURE_LAYER}_pca_variance')

#%% 2. PC1–PC2 散点图 + 四象限代表刺激

N_QUAD = 9         # 每象限代表图数量（须为完全平方数）
QUAD_GRID = 3
assert N_QUAD == QUAD_GRID ** 2

QUAD_ORDER = ('tl', 'tr', 'bl', 'br')
QUADS = dict(
    tl=dict(color='#e8a0b8', score=lambda x, y: -x + y,
            mask=lambda x, y: (x <= 0) & (y >= 0)),   # PC1−, PC2+
    tr=dict(color='#f0b080', score=lambda x, y: x + y,
            mask=lambda x, y: (x >= 0) & (y >= 0)),    # PC1+, PC2+
    bl=dict(color='#98b8e8', score=lambda x, y: -x - y,
            mask=lambda x, y: (x <= 0) & (y <= 0)),  # PC1−, PC2−
    br=dict(color='#98d0a0', score=lambda x, y: x - y,
            mask=lambda x, y: (x >= 0) & (y <= 0)),    # PC1+, PC2−
)

pc1, pc2 = coords[:, 0], coords[:, 1]


def _pick_quad_indices(pc1, pc2, meta, k, exclude):
    """在真实象限内按 score 取 top-k，并排除已被其他象限选中的点。"""
    candidates = np.where(meta['mask'](pc1, pc2) & ~exclude)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int)
    scores = meta['score'](pc1[candidates], pc2[candidates])
    take = min(k, len(candidates))
    order = np.argsort(scores)[-take:][::-1]
    return candidates[order]


exclude = np.zeros(len(pc1), dtype=bool)
quad_idx = {}
for q in QUAD_ORDER:
    quad_idx[q] = _pick_quad_indices(pc1, pc2, QUADS[q], N_QUAD, exclude)
    exclude[quad_idx[q]] = True


def _corner_panel(ax_corner, idxs, color):
    """单角面板：3×3 inset 缩略图 + 彩色外框。"""
    ax_corner.set_xticks([])
    ax_corner.set_yticks([])
    ax_corner.set_xlim(0, 1)
    ax_corner.set_ylim(0, 1)
    pad, gap = 0.03, 0.03
    n = QUAD_GRID
    cell = (1 - 2 * pad - (n - 1) * gap) / n
    for k, ii in enumerate(idxs):
        c, r = divmod(k, n)
        x = pad + c * (cell + gap)
        y = 1 - pad - (r + 1) * cell - r * gap
        ax_in = ax_corner.inset_axes([x, y, cell, cell])
        ax_in.imshow(Image.open(img_paths[int(ii)]))
        ax_in.axis('off')
    for sp in ax_corner.spines.values():
        sp.set_visible(True)
        sp.set_color(color)
        sp.set_linewidth(3.5)


fig = plt.figure(figsize=(9, 9))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 2.4, 1], height_ratios=[1, 2.4, 1],
                      wspace=0.08, hspace=0.08)

ax = fig.add_subplot(gs[1, 1])
ax.scatter(pc1, pc2, s=14, c='0.82', alpha=0.45, edgecolors='none', rasterized=True, zorder=2)

for q in QUAD_ORDER:
    idx = quad_idx[q]
    ax.scatter(pc1[idx], pc2[idx], s=72, c=QUADS[q]['color'], edgecolors='white',
               linewidths=1.0, zorder=4)

ax.axhline(0, color='0.35', lw=0.9, zorder=1)
ax.axvline(0, color='0.35', lw=0.9, zorder=1)
ax.set_xlabel(f'PC1 ({ev_ratio[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({ev_ratio[1]:.1%} var)')
ax.set_title(f'AlexNet {layer_label} object space (NSD1k, N={len(pc1)})')
ax.set_aspect('equal', adjustable='box')

_corner_panel(fig.add_subplot(gs[0, 0]), quad_idx['tl'], QUADS['tl']['color'])
_corner_panel(fig.add_subplot(gs[0, 2]), quad_idx['tr'], QUADS['tr']['color'])
_corner_panel(fig.add_subplot(gs[2, 0]), quad_idx['bl'], QUADS['bl']['color'])
_corner_panel(fig.add_subplot(gs[2, 2]), quad_idx['br'], QUADS['br']['color'])

_save_fig(fig, f'step1_{FEATURE_LAYER}_pc12_schematic')

#%% 3. 将 metamer1k 嵌入样本空间 + PC1–PC2 散点图

N_METAMER = 1000
CACHE2 = savepath + rf'\alexnet_metamer1k_step2_{FEATURE_TAG}.npz'

d1 = np.load(CACHE, allow_pickle=True)
pc_mean, pc_components = d1['pc_mean'], d1['pc_components']
nsd_coords, ev_ratio = d1['coords'], d1['ev_ratio']
layer_label = 'last conv' if FEATURE_LAYER == 'last_conv' else 'fc6'

meta_paths = [ot.Join(metamer_img_path, f'{i:04d}.jpg') for i in range(1, N_METAMER + 1)]
assert os.path.isfile(meta_paths[0]) and os.path.isfile(meta_paths[-1]), 'metamer images not found'

if os.path.isfile(CACHE2):
    d2 = np.load(CACHE2, allow_pickle=True)
    meta_coords = d2['coords']
    print(f'loaded cache: {CACHE2}')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()

    def _extract(batch_tensor):
        if FEATURE_LAYER == 'last_conv':
            buf = []

            def _hook(m, inp, out):
                buf.append(out.detach().cpu().clone())

            h = model.features[10].register_forward_hook(_hook)
            model.features(batch_tensor)
            h.remove()
            return buf[0].flatten(1)
        buf = []

        def _hook(m, inp, out):
            buf.append(out.detach().cpu())

        h = model.classifier[1].register_forward_hook(_hook)
        model(batch_tensor)
        h.remove()
        return buf[0]

    meta_feats = np.zeros((N_METAMER, pc_mean.shape[0]), np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, N_METAMER, BATCH), desc=f'metamer {FEATURE_LAYER}'):
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in meta_paths[i:i + BATCH]]
            meta_feats[i:i + BATCH] = _extract(torch.stack(imgs).to(device)).numpy()

    meta_coords = (meta_feats - pc_mean) @ pc_components.T
    ot.Mkdir(savepath, mute=True)
    np.savez(CACHE2, feats=meta_feats, coords=meta_coords,
             img_paths=np.array(meta_paths, dtype=object), feature_layer=FEATURE_LAYER)
    print(f'saved: {CACHE2}')

print(f'metamer coords: {meta_coords.shape}')

# --- 标签：5×(Raw/S1–S4 × 40)；每块前 20 = ani ---
idx = np.arange(N_METAMER)
within = idx % 200
shuffle = within // 40          # 0=Raw, 1=S1, 2=S2, 3=S3, 4=S4
object_id = within % 40         # 0-39，同一 object 在不同 shuffle level 中保持同色
is_ani = (within % 40) < 20

COLOR_ANI, COLOR_INANI = '#c0392b', '#2980b9'
SHUF_ALPHA = [1.0, 0.72, 0.52, 0.34, 0.18]
STEP2_PC12_COLOR_MODE = 'ani_shuffle'   # 'ani_shuffle' | 'object_rainbow'

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(nsd_coords[:, 0], nsd_coords[:, 1], s=6, c='#dddddd', edgecolors='none',
           zorder=1, rasterized=True)

if STEP2_PC12_COLOR_MODE == 'ani_shuffle':
    for shuf, alpha in enumerate(SHUF_ALPHA):
        for ani_flag, color in ((True, COLOR_ANI), (False, COLOR_INANI)):
            m = (shuffle == shuf) & (is_ani == ani_flag)
            ax.scatter(meta_coords[m, 0], meta_coords[m, 1], s=20, c=color, alpha=alpha,
                       edgecolors='none', zorder=2 + shuf, rasterized=True)
    step2_title_suffix = 'ani/inani color, shuffle alpha'
elif STEP2_PC12_COLOR_MODE == 'object_rainbow':
    object_colors = plt.cm.rainbow(np.linspace(0, 1, 40))
    for obj in range(40):
        m = object_id == obj
        ax.scatter(meta_coords[m, 0], meta_coords[m, 1], s=20, c=[object_colors[obj]], alpha=0.72,
                   edgecolors='none', zorder=2, rasterized=True)
    step2_title_suffix = '40 object rainbow color'
else:
    raise ValueError("STEP2_PC12_COLOR_MODE must be 'ani_shuffle' or 'object_rainbow'")

ax.set_xlabel(f'PC1 ({ev_ratio[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({ev_ratio[1]:.1%} var)')
ax.set_title(f'AlexNet {layer_label} — metamer1k in NSD1k space\n{step2_title_suffix}')
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
_save_fig(fig, f'step2_{FEATURE_LAYER}_metamer_pc12_{STEP2_PC12_COLOR_MODE}')

#%% 4. 拟合 shuffle 二维子空间（50D 正交轴）+ 评估

SHUFFLE_LABELS = ['Raw', 'S1', 'S2', 'S3', 'S4']
SHUFFLE_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

meta50 = meta_coords[:, :N_DIM]
nsd50 = nsd_coords[:, :N_DIM]

# 用每个 shuffle level 的 50D 中心轨迹，定义最能表达 shuffle 变化的 2D 平面。
meta_center = meta50.mean(axis=0)
shuffle_centroids = np.vstack([meta50[shuffle == level].mean(axis=0) for level in range(5)])
shuffle_counts = np.array([(shuffle == level).sum() for level in range(5)])
centroid_centered = shuffle_centroids - meta_center

shuffle_pca = PCA(n_components=2).fit(centroid_centered)
shuffle_axes = shuffle_pca.components_.copy()

# 固定 Axis1 的符号，让它大致指向 Raw → S4，便于跨次运行读图。
if (centroid_centered @ shuffle_axes[0])[-1] < (centroid_centered @ shuffle_axes[0])[0]:
    shuffle_axes[0] *= -1

meta_subspace = (meta50 - meta_center) @ shuffle_axes.T
nsd_subspace = (nsd50 - meta_center) @ shuffle_axes.T
centroid_subspace = centroid_centered @ shuffle_axes.T

print('\n=== Shuffle 2D subspace (centroid trajectory in 50D PCA space) ===')
print('Axis loadings are linear combinations of the first 50 NSD PCs.')
print(f'centroid_trajectory_fit: {shuffle_pca.explained_variance_ratio_.sum():.2%}')

between_50d = np.sum(shuffle_counts[:, None] * centroid_centered ** 2)
between_2d = np.sum(shuffle_counts[:, None] * centroid_subspace ** 2)
between_shuffle_fit = between_2d / between_50d
total_2d = np.sum((meta_subspace - meta_subspace.mean(axis=0)) ** 2)
eta2_2d = between_2d / total_2d

print(f'between_shuffle_fit:     {between_shuffle_fit:.2%}')
print(f'eta2_2d:                 {eta2_2d:.2%}')
print('Level centroids in shuffle subspace:')
for level, label in enumerate(SHUFFLE_LABELS):
    x, y = centroid_subspace[level]
    print(f'  {label:>3s}: Axis1={x:8.3f}, Axis2={y:8.3f}, n={shuffle_counts[level]}')


def _rankdata_average(x):
    """Average ranks for tied values, using 1-based ranks."""
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and sorted_x[stop] == sorted_x[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


dist_to_s4 = np.linalg.norm(meta_subspace - centroid_subspace[4], axis=1)
rho_shuffle_dist_s4 = np.corrcoef(_rankdata_average(shuffle), _rankdata_average(dist_to_s4))[0, 1]
print(f'Spearman shuffle vs distance-to-S4-centroid: {rho_shuffle_dist_s4:.3f}')

STEP4_SUBSPACE_COLOR_MODE = 'object_shuffle'   # 'shuffle_level' | 'object_shuffle'

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(nsd_subspace[:, 0], nsd_subspace[:, 1], s=6, c='#dddddd', edgecolors='none',
           alpha=0.45, zorder=1, rasterized=True, label='NSD1k')

if STEP4_SUBSPACE_COLOR_MODE == 'shuffle_level':
    for level, (label, color) in enumerate(zip(SHUFFLE_LABELS, SHUFFLE_COLORS)):
        m = shuffle == level
        ax.scatter(meta_subspace[m, 0], meta_subspace[m, 1], s=18, c=color, alpha=0.55,
                   edgecolors='none', zorder=2 + level, rasterized=True, label=f'{label} ({m.sum()})')
    step4_title_suffix = 'shuffle level color'
elif STEP4_SUBSPACE_COLOR_MODE == 'object_shuffle':
    object_colors = plt.cm.rainbow(np.linspace(0, 1, 40))
    for shuf, alpha in enumerate(SHUF_ALPHA):
        for obj in range(40):
            m = (shuffle == shuf) & (object_id == obj)
            ax.scatter(meta_subspace[m, 0], meta_subspace[m, 1], s=18, c=[object_colors[obj]],
                       alpha=alpha, edgecolors='none', zorder=2 + shuf, rasterized=True)
    for shuf, (label, alpha) in enumerate(zip(SHUFFLE_LABELS, SHUF_ALPHA)):
        ax.scatter([], [], s=18, c='0.35', alpha=alpha, edgecolors='none', label=f'{label} (α={alpha:.2f})')
    step4_title_suffix = '40 object rainbow color, shuffle alpha'
else:
    raise ValueError("STEP4_SUBSPACE_COLOR_MODE must be 'shuffle_level' or 'object_shuffle'")

ax.plot(centroid_subspace[:, 0], centroid_subspace[:, 1], '-k', lw=1.4, zorder=10)
for level, (label, color) in enumerate(zip(SHUFFLE_LABELS, SHUFFLE_COLORS)):
    ax.scatter(centroid_subspace[level, 0], centroid_subspace[level, 1], s=95, c=color,
               edgecolors='white', linewidths=1.1, zorder=11)
    ax.text(centroid_subspace[level, 0], centroid_subspace[level, 1], f' {label}',
            fontsize=9, weight='bold', va='center', zorder=12)

ax.axhline(0, color='0.35', lw=0.8, zorder=0)
ax.axvline(0, color='0.35', lw=0.8, zorder=0)
ax.set_xlabel(f'Shuffle Axis 1 ({shuffle_pca.explained_variance_ratio_[0]:.1%} centroid var)')
ax.set_ylabel(f'Shuffle Axis 2 ({shuffle_pca.explained_variance_ratio_[1]:.1%} centroid var)')
ax.set_title(f'AlexNet {layer_label} — 2D shuffle subspace\n{step4_title_suffix}')
ax.set_aspect('equal', adjustable='box')
ax.legend(frameon=False, fontsize=8, loc='best')
fig.tight_layout()
_save_fig(fig, f'step4_{FEATURE_LAYER}_shuffle_subspace_{STEP4_SUBSPACE_COLOR_MODE}')

fig, axes = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
pc_ids = np.arange(1, N_DIM + 1)
for ax_i, axis_idx in enumerate(range(2)):
    axes[ax_i].bar(pc_ids, shuffle_axes[axis_idx], color='0.25', width=0.8)
    axes[ax_i].axhline(0, color='0.35', lw=0.8)
    axes[ax_i].set_ylabel(f'Axis {axis_idx + 1}')
    axes[ax_i].set_title(f'Shuffle Axis {axis_idx + 1} loadings on original PCs', fontsize=10)
axes[-1].set_xlabel('Original NSD PCA component')
fig.tight_layout()
_save_fig(fig, f'step4_{FEATURE_LAYER}_shuffle_axis_loadings')

#%% 5. 拟合一根 shuffle 轴，并展示每张图在轴上的 load

# 如果 shuffle 变化基本是一根轴，centroid 轨迹的第一主成分就是最直接的 shuffle axis。
shuffle_axis_1d = PCA(n_components=1).fit(centroid_centered).components_[0].copy()
centroid_load_1d = centroid_centered @ shuffle_axis_1d
if centroid_load_1d[-1] < centroid_load_1d[0]:
    shuffle_axis_1d *= -1
    centroid_load_1d *= -1

meta_load_1d = (meta50 - meta_center) @ shuffle_axis_1d
nsd_load_1d = (nsd50 - meta_center) @ shuffle_axis_1d

between_1d = np.sum(shuffle_counts * centroid_load_1d ** 2)
between_shuffle_fit_1d = between_1d / between_50d
eta2_1d = between_1d / np.sum((meta_load_1d - meta_load_1d.mean()) ** 2)
rho_shuffle_load = np.corrcoef(_rankdata_average(shuffle), _rankdata_average(meta_load_1d))[0, 1]

level_x = np.arange(5)
centroid_fit = np.polyfit(level_x, centroid_load_1d, 1)
centroid_pred = np.polyval(centroid_fit, level_x)
centroid_level_r2 = 1 - np.sum((centroid_load_1d - centroid_pred) ** 2) / np.sum((centroid_load_1d - centroid_load_1d.mean()) ** 2)


def _linear_fit_r2(x, y):
    fit = np.polyfit(x, y, 1)
    pred = np.polyval(fit, x)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / ss_tot
    return fit, pred, r2


ani_level_mean = np.array([meta_load_1d[(shuffle == level) & is_ani].mean() for level in range(5)])
inani_level_mean = np.array([meta_load_1d[(shuffle == level) & ~is_ani].mean() for level in range(5)])
ani_fit, ani_pred, ani_r2 = _linear_fit_r2(level_x, ani_level_mean)
inani_fit, inani_pred, inani_r2 = _linear_fit_r2(level_x, inani_level_mean)

print('\n=== Shuffle 1D axis ===')
print('Axis is PC1 of the 5 shuffle-level centroids in 50D PCA space.')
print(f'between_shuffle_fit_1d: {between_shuffle_fit_1d:.2%}')
print(f'eta2_1d:                {eta2_1d:.2%}')
print(f'centroid_level_R2:      {centroid_level_r2:.2%}')
print(f'ani_level_R2:           {ani_r2:.2%}')
print(f'inani_level_R2:         {inani_r2:.2%}')
print(f'Spearman shuffle vs axis load: {rho_shuffle_load:.3f}')
print(f'NSD1k load range on this axis: {nsd_load_1d.min():.3f} to {nsd_load_1d.max():.3f}')

fig, ax = plt.subplots(figsize=(6.2, 4.2))
box_width = 0.22
ani_pos = level_x - 0.12
inani_pos = level_x + 0.12
box_groups = (
    (ani_pos, COLOR_ANI, 'ani', [meta_load_1d[(shuffle == level) & is_ani] for level in range(5)], ani_pred),
    (inani_pos, COLOR_INANI, 'inani', [meta_load_1d[(shuffle == level) & ~is_ani] for level in range(5)], inani_pred),
)

for pos, color, label, data, pred in box_groups:
    bp = ax.boxplot(data, positions=pos, widths=box_width, whis=(10, 90), patch_artist=True,
                    showfliers=False, manage_ticks=False)
    for patch in bp['boxes']:
        patch.set(facecolor=color, alpha=0.35, edgecolor=color, linewidth=1.2)
    for whisker in bp['whiskers']:
        whisker.set(color=color, linewidth=1.0)
    for cap in bp['caps']:
        cap.set(color=color, linewidth=1.0)
    for median in bp['medians']:
        median.set(color=color, linewidth=1.6)
    ax.plot(level_x, pred, color=color, lw=1.6, label=f'{label} fit')
    ax.scatter(level_x, [d.mean() for d in data], s=28, color=color, edgecolors='white',
               linewidths=0.7, zorder=10, label=f'{label} mean')

ax.axhline(0, color='0.35', lw=0.8)
ax.set_xticks(level_x)
ax.set_xticklabels(SHUFFLE_LABELS)
ax.set_xlabel('Shuffle level')
ax.set_ylabel('Load on shuffle axis')
ax.set_title(f'AlexNet {layer_label} — 1D shuffle axis load (boxplot whis 10-90)')
ax.legend(frameon=False, fontsize=8, loc='best')
fig.tight_layout()
_save_fig(fig, f'step5_{FEATURE_LAYER}_shuffle_axis_load_by_level')

fig, ax = plt.subplots(figsize=(4, 2.6))
ax.bar(pc_ids, shuffle_axis_1d, color='0.25', width=0.8)
ax.axhline(0, color='0.35', lw=0.8)
ax.set_xlabel('Original NSD PCA component')
ax.set_ylabel('Axis loading')
ax.set_title('1D shuffle axis loadings on original PCs')
fig.tight_layout()
_save_fig(fig, f'step5_{FEATURE_LAYER}_shuffle_axis_1d_loadings')

#%% 6. 50 PC × 200 metamer heatmap（5 个 cycle 平均）

meta50_cycle_mean = meta_coords[:, :N_DIM].reshape(5, 200, N_DIM).mean(axis=0)
heatmap_data = meta50_cycle_mean.T

fig, ax = plt.subplots(figsize=(8, 5))
vmax = np.percentile(np.abs(heatmap_data), 98)
im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
               interpolation='nearest')

for x in range(40, 200, 40):
    ax.axvline(x - 0.5, color='k', lw=0.6, alpha=0.45)

ax.set_xticks(np.arange(20, 200, 40) - 0.5)
ax.set_xticklabels(['Raw', 'S1', 'S2', 'S3', 'S4'])
ax.set_yticks(np.arange(0, N_DIM, 5))
ax.set_yticklabels(np.arange(1, N_DIM + 1, 5))
ax.set_xlabel('Metamer image, averaged across 5 cycles')
ax.set_ylabel('PC')
ax.set_title(f'AlexNet {layer_label} — metamer loads on first {N_DIM} PCs')
fig.colorbar(im, ax=ax, label='PC load')
fig.tight_layout()
_save_fig(fig, f'step6_{FEATURE_LAYER}_pc50_metamer200_heatmap')

#%%
