'''
试验是否单个神经元对metamer刺激的响应能否被样本空间所解释
是否单个神经元表征样本空间中特定的一根轴，不同metamer打乱在这个轴上的载荷变化是如何的
'''


#%%

# 图片目录，图片格式可能是.bmp或.jpg
nsd_figpath = r'E:\#Stimsets\NSD1000'
metamer_figpath = r'E:\#Stimsets\Metamer_P4_C4321_Object_STI150_1300'

# 神经活动数据，请参考readme.md了解数据结构
cell_rootpath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Raw_Metamer_1k'
check_area = 'ASB'
# 中间变量的保存路径（共享缓存在根目录；脑区结果在 savepath/{area}/）
savepath = r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis'

from obj_space_paths import area_dir, area_path, resolve_area_path, rsp_path, shared_path
from obj_space_plot import finish_fig, SCRIPT_TEST_RSP

SAVE_FIGURES = True
SHOW_FIGURES = True

#%% 1.建立50D样本空间（使用NSD图片），并返回PCA解释的VAR，以及PC1-2，最喜欢和最不喜欢的10张图片
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

N_DIM = 50          # Bao et al.: fixed 50D object space
BATCH = 32
plot_pc = 3         # 1 or 2: which PC to show extreme images for
n_extreme = 10      # top/bottom images per PC
CACHE = shared_path(savepath, 'step1')

# --- NSD1k image list ---
img_paths = sorted(Path(nsd_figpath).glob('*.bmp')) + sorted(Path(nsd_figpath).glob('*.jpg'))
img_paths = [str(p) for p in img_paths]
assert len(img_paths) == 1000, f'expected 1000 NSD images, got {len(img_paths)}'

if os.path.isfile(CACHE):
    d = np.load(CACHE, allow_pickle=True)
    if int(d['n_dim']) != N_DIM:
        raise ValueError(f'cache n_dim={int(d["n_dim"])}, expected {N_DIM}; delete {CACHE} and rerun')
    fc6 = d['fc6']
    cumvar = d['cumvar']
    n_dim = int(d['n_dim'])
    coords = d['coords']
    pc_mean = d['pc_mean']
    pc_components = d['pc_components']
    img_paths = list(d['img_paths'])
    ev_ratio = d['ev_ratio'] if 'ev_ratio' in d.files else np.diff(np.concatenate([[0], cumvar]))[:n_dim]
    print(f'loaded cache: {CACHE}')
else:
    # --- AlexNet fc6 (4096D), same layer as Bao et al. ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
    buf = []

    def _hook(m, inp, out):
        buf.append(out.detach().cpu())

    h = model.classifier[1].register_forward_hook(_hook)   # fc6 linear, 4096 units
    fc6 = np.zeros((len(img_paths), 4096), np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), BATCH), desc='AlexNet fc6'):
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in img_paths[i:i + BATCH]]
            buf.clear()
            model(torch.stack(imgs).to(device))
            fc6[i:i + BATCH] = buf[0].numpy()
    h.remove()

    # --- PCA: keep first 50 PCs (Bao et al.) ---
    pca_full = PCA().fit(fc6)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ev_ratio = pca_full.explained_variance_ratio_[:N_DIM]
    n_dim = N_DIM
    pc_mean = pca_full.mean_.astype(np.float32)
    pc_components = pca_full.components_[:N_DIM].astype(np.float32)
    coords = pca_full.transform(fc6)[:, :N_DIM]

    ot.Mkdir(savepath, mute=True)
    np.savez(CACHE, fc6=fc6, cumvar=cumvar, ev_ratio=ev_ratio, n_dim=n_dim, coords=coords,
             pc_mean=pc_mean, pc_components=pc_components,
             img_paths=np.array(img_paths, dtype=object))
    print(f'saved: {CACHE}')

print(f'{n_dim}D object space (first {N_DIM} PCs, explain {cumvar[N_DIM - 1]:.1%} fc6 variance)')
print(f'PC1–PC2 explain {cumvar[1]:.1%} variance')

# --- extreme images for chosen PC (1-based: plot_pc = 1 or 2) ---
plot_pc = int(plot_pc)
assert 1 <= plot_pc <= n_dim, f'plot_pc must be in [1, {n_dim}]'
pc_idx = plot_pc - 1
scores = coords[:, pc_idx]

def _extreme(scores, k=10):
    order = np.argsort(scores)
    return order[-k:][::-1], order[:k]

hi_idx, lo_idx = _extreme(scores, n_extreme)

fig, axes = plt.subplots(2, n_extreme, figsize=(1.2 * n_extreme, 2.8))
for col, i in enumerate(hi_idx):
    axes[0, col].imshow(Image.open(img_paths[i]))
    axes[0, col].axis('off')
for col, i in enumerate(lo_idx):
    axes[1, col].imshow(Image.open(img_paths[i]))
    axes[1, col].axis('off')
axes[0, 0].set_ylabel(f'PC{plot_pc} hi', fontsize=9)
axes[1, 0].set_ylabel(f'PC{plot_pc} lo', fontsize=9)
fig.suptitle(f'NSD1k extremes on PC{plot_pc} ({ev_ratio[pc_idx]:.1%} var)', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_TEST_RSP, f'step1_pc{plot_pc}_extremes')

fig, ax = plt.subplots(figsize=(3, 2.5))
ax.plot(np.arange(1, len(cumvar) + 1), cumvar, 'k-', lw=1.5)
ax.axvline(N_DIM, color='C1', ls='--', lw=0.8, label=f'K={N_DIM} ({cumvar[N_DIM - 1]:.1%})')
ax.set_xlim(1, 100)
ax.set_xlabel('N PCs')
ax.set_ylabel('Explained VAR')
ax.legend(fontsize=8)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'step1_pca_variance')

#%% 2.将metamer1300嵌入样本空间，得到样本空间中的坐标
N_METAMER = 1000      # 0001–1000: metamer; 1001–1300: STI/FOB (from NSD1k), skip here
CACHE2 = shared_path(savepath, 'step2')

d1 = np.load(CACHE, allow_pickle=True)
pc_mean = d1['pc_mean']
pc_components = d1['pc_components']
nsd_coords = d1['coords']

meta_paths = [ot.Join(metamer_figpath, f'{i:04d}.jpg') for i in range(1, N_METAMER + 1)]
assert all(os.path.isfile(p) for p in meta_paths[:3] + meta_paths[-3:]), 'metamer images not found'

if os.path.isfile(CACHE2):
    d2 = np.load(CACHE2, allow_pickle=True)
    meta_fc6 = d2['fc6']
    meta_coords = d2['coords']
    meta_paths = list(d2['img_paths'])
    print(f'loaded cache: {CACHE2}')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
    buf = []

    def _hook(m, inp, out):
        buf.append(out.detach().cpu())

    h = model.classifier[1].register_forward_hook(_hook)
    meta_fc6 = np.zeros((N_METAMER, 4096), np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, N_METAMER, BATCH), desc='metamer fc6'):
            imgs = [preprocess(Image.open(p).convert('RGB')) for p in meta_paths[i:i + BATCH]]
            buf.clear()
            model(torch.stack(imgs).to(device))
            meta_fc6[i:i + BATCH] = buf[0].numpy()
    h.remove()

    # project onto NSD1k PCA basis (out-of-sample, same as Bao et al.)
    meta_coords = (meta_fc6 - pc_mean) @ pc_components.T
    ot.Mkdir(savepath, mute=True)
    np.savez(CACHE2, fc6=meta_fc6, coords=meta_coords,
             img_paths=np.array(meta_paths, dtype=object))
    print(f'saved: {CACHE2}')

print(f'metamer coords: {meta_coords.shape}  (50D embedded in NSD1k object space)')

# --- labels: 5 cycles × (Raw/S1–S4 × 40); ani=in first 20 per block ---
idx = np.arange(N_METAMER)
within = idx % 200
shuffle = within // 40          # 0=Raw, 1=S1(C4), 2=S2(C3), 3=S3(C2), 4=S4(C1)
is_ani = (within % 40) < 20
plot_idx = idx < 200            # first cycle only → 200 unique stimuli, no overlap

COLOR_ANI = '#c0392b'
COLOR_INANI = '#2980b9'
SHUF_LABELS = ['Raw', 'S1', 'S2', 'S3', 'S4']
SHUF_ALPHA = [1.0, 0.72, 0.52, 0.34, 0.18]   # Raw darkest → S4 lightest

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(nsd_coords[:, 0], nsd_coords[:, 1], s=6, c='#dddddd', edgecolors='none',
           label='NSD1k', zorder=1, rasterized=True)

for shuf, lab, alpha in zip(range(5), SHUF_LABELS, SHUF_ALPHA):
    for ani_flag, color, tag in ((True, COLOR_ANI, 'Ani'), (False, COLOR_INANI, 'Inani')):
        m = plot_idx & (shuffle == shuf) & (is_ani == ani_flag)
        ax.scatter(meta_coords[m, 0], meta_coords[m, 1], s=22, c=color, alpha=alpha,
                   edgecolors='white', linewidths=0.25, zorder=3 + shuf, rasterized=True)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('NSD1k object space (PC1–PC2)')
ax.grid(True, ls=':', lw=0.6, alpha=0.5)

from matplotlib.lines import Line2D

# --- 40 fitted vectors: linear fit PC1/PC2 ~ shuffle (0=Raw … 4=S4) ---
SHUF_LEV = np.arange(5, dtype=float)
meta_vec = np.zeros((40, 2))   # fitted Raw→S4 displacement in PC1–PC2

for obj in range(40):
    pts = meta_coords[[obj + s * 40 for s in range(5)], :2]
    c = COLOR_ANI if obj < 20 else COLOR_INANI
    m1, m2 = np.polyfit(SHUF_LEV, pts[:, 0], 1), np.polyfit(SHUF_LEV, pts[:, 1], 1)
    org = np.array([m1[1], m2[1]])
    vec = np.array([np.polyval(m1, 4) - org[0], np.polyval(m2, 4) - org[1]])
    meta_vec[obj] = vec
    ax.quiver(org[0], org[1], vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
              color=c, alpha=0.85, width=0.003, headwidth=4, headlength=5, zorder=6)

leg_ani = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ANI, markersize=7, label='Ani'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INANI, markersize=7, label='Inani'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', markersize=6, label='NSD1k'),
    Line2D([0], [0], color='#555555', lw=1.3, alpha=0.85, label='Fitted vector (Raw→S4)'),
]
leg_shuf = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', markersize=7,
           alpha=a, label=lab)
    for lab, a in zip(SHUF_LABELS, SHUF_ALPHA)
]
leg1 = ax.legend(handles=leg_ani, loc='upper left', fontsize=8, framealpha=0.92, title='Category')
ax.add_artist(leg1)
ax.legend(handles=leg_shuf, loc='lower right', fontsize=8, framealpha=0.92, title='Shuffle')
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'step2_metamer_pc12')

#%% 3.拟合每个神经元的50D偏好轴并保存（重跑此 cell 才会重新计算）
import pandas as pd

n_extreme = 10
CACHE3 = area_path(savepath, check_area, 'obj_axis_fit')
SUMMARY_CSV = area_path(savepath, check_area, 'obj_axis_summary')

d1 = np.load(CACHE, allow_pickle=True)
d2 = np.load(CACHE2, allow_pickle=True)
nsd_coords = d1['coords']
meta_coords = d2['coords']
rsp = np.load(rsp_path(cell_rootpath, check_area))
assert meta_coords.shape[0] == rsp.shape[1] == 1000

F_mu = meta_coords.mean(0)
F_std = meta_coords.std(0)
F_std[F_std < 1e-8] = 1.0
F_z = (meta_coords - F_mu) / F_std
X = np.c_[F_z, np.ones(len(F_z))]

n_cell = rsp.shape[0]
axes = np.zeros((n_cell, N_DIM), np.float32)
bias = np.zeros(n_cell, np.float32)
r2 = np.zeros(n_cell, np.float32)
meta_load = np.zeros((n_cell, 1000), np.float32)

for i in range(n_cell):
    coef, _, _, _ = np.linalg.lstsq(X, rsp[i], rcond=None)
    axes[i] = coef[:N_DIM]
    bias[i] = coef[N_DIM]
    meta_load[i] = F_z @ axes[i]
    pred = meta_load[i] + bias[i]
    v = rsp[i].var()
    r2[i] = 1.0 - (rsp[i] - pred).var() / v if v > 0 else np.nan

nsd_F_z = (nsd_coords - F_mu) / F_std
nsd_load_all = nsd_F_z @ axes.T

hi_cols = [f'nsd_hi_{k}' for k in range(1, n_extreme + 1)]
lo_cols = [f'nsd_lo_{k}' for k in range(1, n_extreme + 1)]
summary_rows = []
for i in range(n_cell):
    order = np.argsort(nsd_load_all[:, i])
    row = {'cell_idx': i, 'r2': r2[i]}
    for k, j in enumerate(order[-n_extreme:][::-1]):
        row[hi_cols[k]] = int(j)
    for k, j in enumerate(order[:n_extreme]):
        row[lo_cols[k]] = int(j)
    summary_rows.append(row)

ot.Mkdir(area_dir(savepath, check_area), mute=True)
np.savez(CACHE3, axes=axes, bias=bias, r2=r2, meta_load=meta_load,
         F_mu=F_mu, F_std=F_std, check_area=np.array(check_area))
pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

print(f'{check_area}: {n_cell} cells, median R² = {np.nanmedian(r2):.3f}, mean R² = {np.nanmean(r2):.3f}')
print(f'saved: {CACHE3}')
print(f'saved: {SUMMARY_CSV}')

#%% 3plot.单细胞可视化（只读缓存，改 plot_cell 后重跑本 cell 即可）
from matplotlib.patches import ConnectionPatch

plot_cell = 780
n_extreme = 10
n_meta_plot = 5

d1 = np.load(CACHE, allow_pickle=True)
d2 = np.load(CACHE2, allow_pickle=True)
d3 = np.load(resolve_area_path(savepath, check_area, 'obj_axis_fit'), allow_pickle=True)
nsd_img_paths = list(d1['img_paths'])
meta_paths = list(d2['img_paths'])
nsd_coords = d1['coords']
F_mu, F_std = d3['F_mu'], d3['F_std']
axes_fit = d3['axes']
r2 = d3['r2']
meta_load = d3['meta_load']
rsp = np.load(rsp_path(cell_rootpath, check_area))

pc = int(plot_cell)
nsd_load = ((nsd_coords - F_mu) / F_std) @ axes_fit[pc]
order = np.argsort(nsd_load)
hi_idx, lo_idx = order[-n_extreme:][::-1], order[:n_extreme]
print(f'{check_area} cell {pc}: R²={r2[pc]:.3f}')

# --- NSD extremes ---
fig, axes_img = plt.subplots(2, n_extreme, figsize=(1.2 * n_extreme, 2.8))
for col, j in enumerate(hi_idx):
    axes_img[0, col].imshow(Image.open(nsd_img_paths[j]))
    axes_img[0, col].axis('off')
for col, j in enumerate(lo_idx):
    axes_img[1, col].imshow(Image.open(nsd_img_paths[j]))
    axes_img[1, col].axis('off')
axes_img[0, 0].set_ylabel('axis hi', fontsize=9)
axes_img[1, 0].set_ylabel('axis lo', fontsize=9)
fig.suptitle(f'{check_area} cell {pc} — NSD extremes (R²={r2[pc]:.3f})', fontsize=10)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_TEST_RSP, f'cell{pc}_nsd_extremes', area=check_area)

# --- metamer tuning + side thumbnails (vertical) ---
x = meta_load[pc]
y = rsp[pc]
xl = np.linspace(x.min(), x.max(), 100)
m, b = np.polyfit(x, y, 1)
load_u = meta_load[pc].reshape(5, 200).mean(0)
hi_stim = np.argsort(load_u)[-n_meta_plot:][::-1]
lo_stim = np.argsort(load_u)[:n_meta_plot]

fig = plt.figure(figsize=(7.5, 5.5))
gs = fig.add_gridspec(n_meta_plot, 3, width_ratios=[1, 3.2, 1], hspace=0.12, wspace=0.08)
ax = fig.add_subplot(gs[:, 1])
axes_lo = [fig.add_subplot(gs[i, 0]) for i in range(n_meta_plot)]
axes_hi = [fig.add_subplot(gs[i, 2]) for i in range(n_meta_plot)]

ax.scatter(x, y, s=10, alpha=0.28, c='0.78', edgecolors='none', zorder=1)
ax.plot(xl, m * xl + b, 'r-', lw=1.5, zorder=2)
ax.scatter(x[hi_stim], y[hi_stim], s=36, facecolors='none', edgecolors='#c0392b', lw=1.2, zorder=4)
ax.scatter(x[lo_stim], y[lo_stim], s=36, facecolors='none', edgecolors='#2980b9', lw=1.2, zorder=4)

for i, stim in enumerate(lo_stim):
    axes_lo[i].imshow(Image.open(meta_paths[stim]))
    axes_lo[i].set_xticks([])
    axes_lo[i].set_yticks([])
    for sp in axes_lo[i].spines.values():
        sp.set_edgecolor('#2980b9')
        sp.set_linewidth(1.2)
    fig.add_artist(ConnectionPatch(
        (1.0, 0.5), (x[stim], y[stim]), 'axes fraction', 'data',
        axesA=axes_lo[i], axesB=ax, color='#2980b9', lw=0.7, alpha=0.55, zorder=3,
    ))
for i, stim in enumerate(hi_stim):
    axes_hi[i].imshow(Image.open(meta_paths[stim]))
    axes_hi[i].set_xticks([])
    axes_hi[i].set_yticks([])
    for sp in axes_hi[i].spines.values():
        sp.set_edgecolor('#c0392b')
        sp.set_linewidth(1.2)
    fig.add_artist(ConnectionPatch(
        (0.0, 0.5), (x[stim], y[stim]), 'axes fraction', 'data',
        axesA=axes_hi[i], axesB=ax, color='#c0392b', lw=0.7, alpha=0.55, zorder=3,
    ))

ax.set_xlabel('Loading on preferred axis')
ax.set_ylabel('Response (spikes)')
ax.set_title(f'{check_area} cell {pc}  R²={r2[pc]:.3f}  (unique metamer hi/lo ×{n_meta_plot})')
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_TEST_RSP, f'cell{pc}_metamer_tuning', area=check_area)

#%% 4.群体统计：R² 分布 & 神经元偏好轴两两夹角（只读 obj_axis_fit）
d3 = np.load(resolve_area_path(savepath, check_area, 'obj_axis_fit'), allow_pickle=True)
r2_pop = d3['r2']
axes_pop = d3['axes']
n_pop = len(r2_pop)
r2_valid = r2_pop[np.isfinite(r2_pop)]

# --- fig1: R² across all cells ---
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.hist(r2_valid, bins=30, color='0.55', edgecolor='white', lw=0.6)
med = np.median(r2_valid)
ax.axvline(med, color='C1', ls='--', lw=1.5, label=f'median = {med:.3f}')
ax.axvline(np.mean(r2_valid), color='C0', ls=':', lw=1.2, label=f'mean = {np.mean(r2_valid):.3f}')
ax.set_xlabel('R² (50D axis model)')
ax.set_ylabel('Cell count')
ax.set_title(f'{check_area}  N = {n_pop}')
ax.legend(fontsize=8)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'pop_r2_hist', area=check_area)

# --- fig2: pairwise axis angles in 50D (degrees) ---
u = axes_pop / (np.linalg.norm(axes_pop, axis=1, keepdims=True) + 1e-8)
cos_mat = np.clip(u @ u.T, -1.0, 1.0)
angle_deg = np.degrees(np.arccos(cos_mat))

fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(angle_deg, cmap='viridis', vmin=0, vmax=90, origin='lower', aspect='equal')
ax.set_xlabel('Cell index')
ax.set_ylabel('Cell index')
ax.set_title(f'{check_area}  pairwise axis angle ({n_pop}×{n_pop})')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Angle (°)')
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'pop_axis_angle_matrix', area=check_area)

print(f'{check_area}: angle median = {np.median(angle_deg[~np.eye(n_pop, dtype=bool)]):.1f}°')

#%% 5demo.单细胞：1000 张 metamer 轴载荷 vs 神经响应（只读缓存）
from matplotlib.lines import Line2D

demo_cell = 300
N_METAMER = 1000
# 高亮 parent id（每个 shuffle 块内 1–40：1–20 ani，21–40 inani）；不设则全部按色系上色
highlight_ids = [3,23,38]         # 例: [3, 7, 21]
highlight_range = None        # 例: (1, 5) 闭区间，与 highlight_ids 可同时用（取并集）

d3 = np.load(resolve_area_path(savepath, check_area, 'obj_axis_fit'), allow_pickle=True)
rsp = np.load(rsp_path(cell_rootpath, check_area))

pc = int(demo_cell)
x = d3['meta_load'][pc]
y = rsp[pc]
r2_demo = float(d3['r2'][pc])

idx = np.arange(N_METAMER)
within = idx % 200
shuffle = within // 40
is_ani = (within % 40) < 20
parent_id = (within % 40) + 1   # 1–40

hl_set = set()
if highlight_ids is not None:
    hl_set.update(highlight_ids)
if highlight_range is not None:
    hl_set.update(range(int(highlight_range[0]), int(highlight_range[1]) + 1))
use_highlight = len(hl_set) > 0
is_hl = np.isin(parent_id, list(hl_set)) if use_highlight else np.ones(N_METAMER, dtype=bool)

COLOR_ANI = '#c0392b'
COLOR_INANI = '#2980b9'
SHUF_LABELS = ['Raw', 'S1', 'S2', 'S3', 'S4']
SHUF_ALPHA = [1.0, 0.62, 0.38, 0.20, 0.07]
# SHUF_SIZE = [10,10,10,10,10]
SHUF_SIZE = [15]*5

fig, ax = plt.subplots(figsize=(6, 5))
if use_highlight:
    ax.scatter(x[~is_hl], y[~is_hl], s=10, c='0.82', alpha=0.22, edgecolors='none', zorder=1)

for shuf in range(5):
    for ani_flag, color in ((True, COLOR_ANI), (False, COLOR_INANI)):
        m = (shuffle == shuf) & (is_ani == ani_flag) & is_hl
        ax.scatter(x[m], y[m], s=SHUF_SIZE[shuf], c=color, alpha=SHUF_ALPHA[shuf],
                   edgecolors=None, linewidths=0.35, zorder=3 + shuf)

xl = np.linspace(x.min(), x.max(), 100)
m, b = np.polyfit(x, y, 1)
ax.plot(xl, m * xl + b, 'k-', lw=1.5, zorder=2, alpha=0.7)

ax.set_xlabel('Loading on preferred axis')
ax.set_ylabel('Response (spikes)')
hl_note = f'  highlight id={sorted(hl_set)}' if use_highlight else ''
ax.set_title(f'{check_area} cell {pc}  R²={r2_demo:.3f}  (N={N_METAMER}){hl_note}')
ax.grid(True, ls=':', lw=0.5, alpha=0.4)

leg_cat = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ANI, markersize=7, label='Ani'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INANI, markersize=7, label='Inani'),
]
leg_shuf = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
           markersize=SHUF_SIZE[s] ** 0.45, alpha=SHUF_ALPHA[s], label=SHUF_LABELS[s])
    for s in range(5)
]
leg1 = ax.legend(handles=leg_cat, loc='upper left', fontsize=7, title='Category')
ax.add_artist(leg1)
ax.legend(handles=leg_shuf, loc='lower right', fontsize=7, title='Shuffle')
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_TEST_RSP, f'demo_cell{pc}_load_rsp', area=check_area)

#%% 6.Fig3c 风格：每行一神经元，横轴为偏好轴载荷，颜色为归一化发放率（只读缓存）
N_BINS = 40
show_cells = None      # None=全部；或 int，如 150（取排序后前 N）
sort_cells = 'r2'    # 'slope' | 'r2' | 'index'
norm_mode = 'zscore'    # 'cell_minmax' | 'global_p95' | 'p95' | 'zscore'
cell_norm_pct = (5,95)     # 单 cell 原始响应 0-1 归一化范围；可改为 (5, 95) 抗 outlier

d3 = np.load(resolve_area_path(savepath, check_area, 'obj_axis_fit'), allow_pickle=True)
meta_load = d3['meta_load']
axes_fit = d3['axes']
r2_all = d3['r2']
rsp_all = np.load(rsp_path(cell_rootpath, check_area))
n_cell = meta_load.shape[0]

# meta_load = F_z @ axis predicts response; divide by ||axis|| to recover geometric
# distance along each cell's unit preferred axis.
axis_norm = np.linalg.norm(axes_fit, axis=1)
valid_axis = axis_norm > 1e-8
unit_load = np.full_like(meta_load, np.nan, dtype=np.float32)
unit_load[valid_axis] = meta_load[valid_axis] / axis_norm[valid_axis, None]

# Normalize before binning, so the fitted intercept is preserved in the heatmap.
rsp_plot = rsp_all.astype(np.float32).copy()
cell_base = np.full(n_cell, np.nan, dtype=np.float32)
cell_peak = np.full(n_cell, np.nan, dtype=np.float32)
if norm_mode == 'cell_minmax':
    for i in range(n_cell):
        lo, hi = np.nanpercentile(rsp_all[i], cell_norm_pct)
        cell_base[i], cell_peak[i] = lo, hi
        if hi - lo > 1e-8:
            rsp_plot[i] = np.clip((rsp_all[i] - lo) / (hi - lo), 0, 1)

# ramp slope/intercept: displayed response ~ geometric loading on preferred axis
slopes = np.full(n_cell, np.nan, dtype=np.float32)
intercepts = np.full(n_cell, np.nan, dtype=np.float32)
for i in range(n_cell):
    if valid_axis[i]:
        slopes[i], intercepts[i] = np.polyfit(unit_load[i], rsp_plot[i], 1)

print('axis_norm p5/50/95 =', np.nanpercentile(axis_norm, [5, 50, 95]))
print('unit-axis display slope p5/50/95 =', np.nanpercentile(slopes, [5, 50, 95]))
print('unit-axis display intercept p5/50/95 =', np.nanpercentile(intercepts, [5, 50, 95]))

hm = np.full((n_cell, N_BINS), np.nan, np.float32)
edges = np.linspace(-1, 1, N_BINS + 1)

for i in range(n_cell):
    if not valid_axis[i]:
        continue
    lo, hi = np.percentile(unit_load[i], [2.5, 97.5])
    sc = np.clip(2 * (unit_load[i] - lo) / (hi - lo + 1e-8) - 1, -1, 1)
    for b in range(N_BINS):
        m = (sc >= edges[b]) & (sc < edges[b + 1]) if b < N_BINS - 1 else (sc >= edges[b]) & (sc <= edges[b + 1])
        if m.any():
            hm[i, b] = rsp_plot[i, m].mean()

if norm_mode == 'cell_minmax':
    pass
elif norm_mode == 'p95':
    for i in range(n_cell):
        if not np.isfinite(hm[i]).any():
            continue
        denom = np.nanpercentile(hm[i], 95)
        if denom > 1e-8:
            hm[i] = np.clip(hm[i] / denom, 0, 1)
elif norm_mode == 'global_p95':
    global_denom = np.nanpercentile(hm, 95)
    print(f'global p95 = {global_denom:.3f}')
    if global_denom > 1e-8:
        hm = np.clip(hm / global_denom, 0, 1)
elif norm_mode == 'zscore':
    for i in range(n_cell):
        mu, sd = np.nanmean(hm[i]), np.nanstd(hm[i])
        if sd > 1e-8:
            hm[i] = (hm[i] - mu) / sd

if sort_cells == 'slope':
    order = np.argsort(np.nan_to_num(slopes, nan=-np.inf))[::-1]
elif sort_cells == 'r2':
    order = np.argsort(np.nan_to_num(r2_all, nan=-np.inf))[::-1]
elif sort_cells == 'index':
    order = np.arange(n_cell)
else:
    order = np.arange(n_cell)
hm = hm[order]

if show_cells is not None:
    hm = hm[:int(show_cells)]

if norm_mode in ('cell_minmax', 'p95', 'global_p95'):
    cmap, vmin, vmax = 'Reds', 0, 1
    if norm_mode == 'cell_minmax':
        cbar_label = f'Norm. resp. (cell {cell_norm_pct[0]}–{cell_norm_pct[1]}%, before binning)'
    elif norm_mode == 'global_p95':
        cbar_label = 'Norm. resp. (÷ global p95, clipped 0–1)'
    else:
        cbar_label = 'Norm. resp. (÷ cell p95 after binning, clipped 0–1)'
else:
    cmap, vmin, vmax = 'RdBu_r', -2, 2
    cbar_label = 'Norm. resp. (z per cell)'

fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(hm, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', extent=[-1, 1, -0.5, hm.shape[0] - 0.5])
ax.set_xlim(-1, 1)
ax.set_xlabel('Distance along preferred axis  ([−1, 1] = 95% stimuli)')
ylab = {'slope': 'Cell (sorted by ramp slope)', 'r2': 'Cell (sorted by R²)', 'index': 'Cell'}
ax.set_ylabel(ylab.get(sort_cells, 'Cell'))
ax.set_title(f'{check_area}  ramp tuning  N={hm.shape[0]}  norm={norm_mode}')
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label(cbar_label)
fig.tight_layout()
finish_fig(fig, savepath, SCRIPT_TEST_RSP, 'pop_ramp_heatmap', area=check_area)
#%%

