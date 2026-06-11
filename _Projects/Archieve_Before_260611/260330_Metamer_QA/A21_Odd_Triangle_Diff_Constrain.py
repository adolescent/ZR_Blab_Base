'''
Odd-1-out 三角形：Raw–C1..C4（Pearson 距离 = 1−r，不做周长归一化）。

- 每个子图对应一个 (Network, Layer)，在同一坐标系中叠加 Raw–C1..C4 四条轮廓，便于比较约束等级的影响。
- 边长使用 Pearson 距离：d = 1 − r，故 r 越大距离越小（与 A11/A12 等生成 Constrain_Corr 时 Dist=1−Corr 一致）。
'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex
from tqdm import tqdm
import OS_Tools as ot

# --- paths ---
datafolder = r'E:\#Preprocessed_Data\260305_Report_Data\Site_Constrain_Corr'

# 只绘制这些 (Network, Layer)；请按实际 parquet 中的名字填写
SELECTED_NET_LAYERS = [
    ('MSB', 'MSB'),
    ('Alexnet', 'last_conv'),
    ('Alexnet', 'fc6'),
]

# 是否写出完整 triangle 表（较大）；False 时仅在内存中聚合绘图
SAVE_TRIANGLE_PARQUET = False
OUT_PARQUET_NAME = 'Triangle_Table_Raw_C1_C4.parquet'

# %% load & preprocess (same as A21_Odd_One_Triangle)
Constrain_Corr = pd.read_parquet(ot.Join(datafolder, 'Constrain_Corr.parquet')).copy()
Constrain_Corr['V_img1'] = Constrain_Corr['C_img1'].astype(int)
Constrain_Corr['V_img2'] = Constrain_Corr['C_img2'].astype(int)
Constrain_Corr['C_img1'] = Constrain_Corr['V_img1'] % 5
Constrain_Corr['C_img2'] = Constrain_Corr['V_img2'] % 5

# Pearson 距离：应为 1 − Pearson r（相关越高，距离越小）
_corr = pd.to_numeric(Constrain_Corr['Corr'], errors='coerce')
_dist_col = pd.to_numeric(Constrain_Corr['Dist'], errors='coerce')
_delta = np.nanmax(np.abs(_dist_col - (1.0 - _corr)))
print(f'[Pearson distance check] max |Dist − (1−Corr)| in Constrain_Corr: {_delta:.3e}')
if _delta > 1e-4:
    print(
        '  Warning: parquet Dist 与 1−Corr 不一致；三角形边长将统一用 1−Corr 由 Corr 计算。'
    )

v0_list = [0, 5, 10, 15, 20]
group_cols = ['Network', 'Layer', 'Img_Index']

cols = [
    'Network', 'Layer', 'Img_Index', 'constraint_k',
    'C_R1', 'C_R2', 'CC', 'D_R1', 'D_R2', 'D_CC',
]
triangle_table = pd.DataFrame(index=range(1000000), columns=cols)
row_i = 0

for k in range(1, 5):
    sub_mask = (
        ((Constrain_Corr['C_img1'] == 0) & (Constrain_Corr['C_img2'] == k))
        | ((Constrain_Corr['C_img1'] == k) & (Constrain_Corr['C_img2'] == 0))
        | ((Constrain_Corr['C_img1'] == k) & (Constrain_Corr['C_img2'] == k))
    )
    pair_df = Constrain_Corr.loc[
        sub_mask,
        ['Network', 'Layer', 'Img_Index', 'V_img1', 'V_img2', 'Corr', 'Dist', 'C_img1', 'C_img2'],
    ].copy()

    vk_list = [k, k + 5, k + 10, k + 15, k + 20]

    for (net, layer, img_idx), g in tqdm(
        pair_df.groupby(group_cols),
        total=pair_df.groupby(group_cols).ngroups,
        desc=f'constraint_k={k}',
    ):
        v1 = g['V_img1'].to_numpy()
        v2 = g['V_img2'].to_numpy()
        a = np.minimum(v1, v2).astype(int)
        b = np.maximum(v1, v2).astype(int)
        corr = g['Corr'].to_numpy(dtype=np.float64)
        # 显式用 Pearson 距离 1−r，保证「相关越高 → 边长越短」
        dist = 1.0 - corr
        pair_map = {(int(ai), int(bi)): (float(ci), float(di)) for ai, bi, ci, di in zip(a, b, corr, dist)}

        def get_pair(x, y):
            kk = (int(x), int(y)) if x < y else (int(y), int(x))
            return pair_map.get(kk, (np.nan, np.nan))

        for i in range(len(vk_list) - 1):
            vka = vk_list[i]
            for j in range(i + 1, len(vk_list)):
                vkb = vk_list[j]
                c_cc, d_cc = get_pair(vka, vkb)
                for v0 in v0_list:
                    c_r1, d_r1 = get_pair(vka, v0)
                    c_r2, d_r2 = get_pair(vkb, v0)
                    triangle_table.iloc[row_i] = [
                        net, layer, img_idx, k,
                        c_r1, c_r2, c_cc, d_r1, d_r2, d_cc,
                    ]
                    row_i += 1
                    if row_i >= len(triangle_table):
                        triangle_table = pd.concat(
                            [
                                triangle_table,
                                pd.DataFrame(index=range(1000000), columns=triangle_table.columns),
                            ],
                            axis=0,
                        )

triangle_table = triangle_table.iloc[:row_i].reset_index(drop=True)
for c in ['C_R1', 'C_R2', 'CC', 'D_R1', 'D_R2', 'D_CC']:
    triangle_table[c] = pd.to_numeric(triangle_table[c], errors='coerce')
triangle_table['constraint_k'] = triangle_table['constraint_k'].astype(int)

if SAVE_TRIANGLE_PARQUET:
    triangle_table.to_parquet(ot.Join(datafolder, OUT_PARQUET_NAME))

# %% aggregate: mean Pearson distances (no perimeter normalization)
tri_plot = triangle_table.copy()
tri_plot['D_leg'] = (tri_plot['D_R1'] + tri_plot['D_R2']) / 2.0

tri_agg = (
    tri_plot.groupby(['Network', 'Layer', 'constraint_k'], dropna=False, as_index=False)
    .agg(D_CC=('D_CC', 'mean'), D_leg=('D_leg', 'mean'), N=('D_CC', 'size'))
    .reset_index(drop=True)
)

# %% optional: verify k=4 aggregation matches filtering-only k=4 from same table
_check = (
    tri_plot[tri_plot['constraint_k'] == 4]
    .groupby(['Network', 'Layer'], dropna=False, as_index=False)
    .agg(D_CC=('D_CC', 'mean'), D_leg=('D_leg', 'mean'))
)
_merged = tri_agg[tri_agg['constraint_k'] == 4].merge(
    _check, on=['Network', 'Layer'], suffixes=('', '_ref'), how='inner'
)
if len(_merged):
    d_cc_diff = np.nanmax(np.abs(_merged['D_CC'].to_numpy() - _merged['D_CC_ref'].to_numpy()))
    d_leg_diff = np.nanmax(np.abs(_merged['D_leg'].to_numpy() - _merged['D_leg_ref'].to_numpy()))
    print(f'[verify k=4] max |D_CC - ref|: {d_cc_diff:g}, max |D_leg - ref|: {d_leg_diff:g}')
    assert d_cc_diff < 1e-9 and d_leg_diff < 1e-9
else:
    print('[verify k=4] skipped: no constraint_k==4 rows')

# %% triangle geometry (same as A21; inputs are raw mean distances)
def triangle_vertices_upside_down(base, leg):
    half_base = base / 2.0
    try:
        vertical = np.sqrt(np.clip(leg ** 2 - half_base ** 2, 0, None))
    except Exception:
        return None
    x = np.array([0, -half_base, half_base, 0])
    y = np.array([0, vertical, vertical, 0])
    return x, y


def _bounds_from_vertices(xy_list):
    if not xy_list:
        return (-0.5, 0.5), (-0.05, 0.95)
    xs = np.concatenate([x for x, _ in xy_list])
    ys = np.concatenate([y for _, y in xy_list])
    pad_x = 0.05 * (np.nanmax(xs) - np.nanmin(xs) + 1e-9)
    pad_y = 0.05 * (np.nanmax(ys) - np.nanmin(ys) + 1e-9)
    return (
        (float(np.nanmin(xs) - pad_x), float(np.nanmax(xs) + pad_x)),
        (float(np.nanmin(ys) - pad_y), float(np.nanmax(ys) + pad_y)),
    )


# filter selected networks
selected_set = set(SELECTED_NET_LAYERS)
plot_df = tri_agg[
    tri_agg.apply(lambda r: (r['Network'], r['Layer']) in selected_set, axis=1)
].copy()

_present = set(zip(tri_agg['Network'], tri_agg['Layer']))
for p in SELECTED_NET_LAYERS:
    if p not in _present:
        print(f'Warning: no data for selected (Network, Layer)={p}')

# 同一子图内：按 constraint_k 着色，比较不同约束等级
k_order = [4, 3, 2, 1]
k_labels = {4: 'Raw-S4-S4', 3: 'Raw-S3-S3', 2: 'Raw-S2-S2', 1: 'Raw-S1-S1'}
k_colors = sns.color_palette('viridis', n_colors=4)
k_color_map = {kk: to_hex(k_colors[i]) for i, kk in enumerate(k_order)}

n_panels = len(SELECTED_NET_LAYERS)
ncols = min(n_panels, 4)
nrows = int(np.ceil(n_panels / ncols))
fig_w = max(4.0 * ncols, 5.0)
fig_h = max(4.2 * nrows, 4.2)
fig, axes = plt.subplots(nrows, ncols, dpi=240, figsize=(fig_w, fig_h), squeeze=False)

# 先收集所有将要绘制的三角形顶点，用于统一坐标范围与刻度
panel_vertices = {}
global_vertices = []
for net, layer in SELECTED_NET_LAYERS:
    sub = plot_df[(plot_df['Network'] == net) & (plot_df['Layer'] == layer)]
    curr = []
    for k in k_order:
        rows_k = sub[sub['constraint_k'] == k]
        if rows_k.empty:
            continue
        row = rows_k.iloc[0]
        verts = triangle_vertices_upside_down(row['D_CC'], row['D_leg'])
        if verts is None:
            continue
        curr.append((k, verts))
        global_vertices.append(verts)
    panel_vertices[(net, layer)] = curr

# 所有子图共享同一正方形坐标空间和相同总刻度
if global_vertices:
    xs = np.concatenate([x for x, _ in global_vertices])
    ys = np.concatenate([y for _, y in global_vertices])
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
else:
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.05, 0.95

cx = 0.5 * (x_min + x_max)
cy = 0.5 * (y_min + y_max)
span = max(x_max - x_min, y_max - y_min, 1e-6)
span *= 1.10  # global padding
half = 0.5 * span
global_xlim = (cx - half, cx + half)
global_ylim = (cy - half, cy + half)
# 使用原始距离值刻度（不做缩放）
fixed_xticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=float)
fixed_yticks = np.array([-0.1, 0.0, 0.2, 0.4, 0.6, 0.7], dtype=float)

# 统一三图坐标空间：至少覆盖你指定刻度，也覆盖真实数据范围
global_xlim = (
    min(global_xlim[0], float(fixed_xticks.min())),
    max(global_xlim[1], float(fixed_xticks.max())),
)
global_ylim = (
    min(global_ylim[0], float(fixed_yticks.min())),
    max(global_ylim[1], float(fixed_yticks.max())),
)

for idx, (net, layer) in enumerate(SELECTED_NET_LAYERS):
    r, c = divmod(idx, ncols)
    ax = axes[r][c]
    for k, verts in panel_vertices.get((net, layer), []):
        x, y = verts
        color = k_color_map[k]
        ax.plot(x, y, color=color, lw=1.8, alpha=0.95, label=k_labels[k])

    ax.axhline(0, color='k', lw=0.8, alpha=0.25)
    ax.axvline(0, color='k', lw=0.8, alpha=0.25)
    ax.set_box_aspect(1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'{net} — {layer}', fontsize=11, pad=4)
    ax.set_xlim(global_xlim)
    ax.set_ylim(global_ylim)
    ax.set_xticks(fixed_xticks)
    ax.set_yticks(fixed_yticks)
    ax.tick_params(labelsize=8)

# hide empty axes
for j in range(n_panels, nrows * ncols):
    r, c = divmod(j, ncols)
    axes[r][c].set_visible(False)

fig.suptitle('Odd-1-out triangles', fontsize=13, y=0.98)
legend_handles = [
    plt.Line2D([0], [0], color=k_color_map[k], lw=1.8, label=k_labels[k]) for k in k_order
]
fig.legend(
    handles=legend_handles,
    labels=[k_labels[k] for k in k_order],
    title='Constraint',
    frameon=False,
    fontsize=8,
    loc='upper right',
    bbox_to_anchor=(1.05, 0.8),
)
fig.tight_layout(rect=[0.02, 0.04, 0.96, 0.94])
plt.show()
