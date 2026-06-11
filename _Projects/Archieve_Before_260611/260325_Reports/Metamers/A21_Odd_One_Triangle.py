'''
比较不同网络的Odd One Out表现，

拼合各个脑区和各个网络的data，得到大的corr

然后比较
'''

#%%
import seaborn as sns
import OS_Tools as ot
from PIL import Image
import numpy as np
from Matrix_Tools import Corr_Matrix
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

datafolder=r'E:\#Preprocessed_Data\260305_Report_Data\Site_Constrain_Corr'
Constrain_Corr = pd.read_parquet(ot.Join(datafolder,'Constrain_Corr.parquet'))
Constrain_Corr = Constrain_Corr.copy()
# keep original variant indices (0-24), and map to constraint level (0-4)
Constrain_Corr['V_img1'] = Constrain_Corr['C_img1'].astype(int)
Constrain_Corr['V_img2'] = Constrain_Corr['C_img2'].astype(int)
Constrain_Corr['C_img1'] = Constrain_Corr['V_img1'] % 5
Constrain_Corr['C_img2'] = Constrain_Corr['V_img2'] % 5


neuro_sites = ['AL','ASB','ML','MSB']
dcnn_sites = ['Alexnet','Res50','VGG16','Alexnet','Res50','VGG16']
dcnn_layers = ['fc6','avgpool','fc1','last_conv','last_conv','last_conv']

#%% ############### 1.Generate Odd One Out Data ####################
triangle_table = pd.DataFrame(index=range(1000000),columns=['Network','Layer','Img_Index','C_R1','C_R2','CC','D_R1','D_R2','D_CC'])

#%%
################ Odd-1-out triangle (2x4 + 1x0) ################
# keep only constraint-pair types: (0,4), (4,0), (4,4)
sub_mask = (
    ((Constrain_Corr['C_img1'] == 0) & (Constrain_Corr['C_img2'] == 4)) |
    ((Constrain_Corr['C_img1'] == 4) & (Constrain_Corr['C_img2'] == 0)) |
    ((Constrain_Corr['C_img1'] == 4) & (Constrain_Corr['C_img2'] == 4))
)
pair_df = Constrain_Corr.loc[sub_mask, ['Network','Layer','Img_Index','V_img1','V_img2','Corr','Dist','C_img1','C_img2']].copy()

v4_list = [4, 9, 14, 19, 24]
v0_list = [0, 5, 10, 15, 20]

row_i = 0
group_cols = ['Network', 'Layer', 'Img_Index']
for (net, layer, img_idx), g in tqdm(pair_df.groupby(group_cols), total=pair_df.groupby(group_cols).ngroups):
    # build lookup: (min(v1,v2), max(v1,v2)) -> (corr, dist)
    v1 = g['V_img1'].to_numpy()
    v2 = g['V_img2'].to_numpy()
    a = np.minimum(v1, v2).astype(int)
    b = np.maximum(v1, v2).astype(int)
    corr = g['Corr'].to_numpy()
    dist = g['Dist'].to_numpy()
    pair_map = {(int(ai), int(bi)): (float(ci), float(di)) for ai, bi, ci, di in zip(a, b, corr, dist)}

    def get_pair(x, y):
        k = (int(x), int(y)) if x < y else (int(y), int(x))
        return pair_map.get(k, (np.nan, np.nan))

    # enumerate triangles: choose 2 distinct 4-variants and 1 distinct 0-variant
    for i in range(len(v4_list) - 1):
        v4a = v4_list[i]
        for j in range(i + 1, len(v4_list)):
            v4b = v4_list[j]
            c_cc, d_cc = get_pair(v4a, v4b)  # 4-4 edge
            for v0 in v0_list:
                c_r1, d_r1 = get_pair(v4a, v0)  # 4-0 edge
                c_r2, d_r2 = get_pair(v4b, v0)  # 4-0 edge

                triangle_table.iloc[row_i] = [net, layer, img_idx, c_r1, c_r2, c_cc, d_r1, d_r2, d_cc]
                row_i += 1
                if row_i >= len(triangle_table):
                    # extend if needed
                    triangle_table = pd.concat(
                        [triangle_table, pd.DataFrame(index=range(1000000), columns=triangle_table.columns)],
                        axis=0
                    )

triangle_table = triangle_table.iloc[:row_i].reset_index(drop=True)

triangle_table['Selective_Index'] = (triangle_table['D_R1']+triangle_table['D_R2']-2*triangle_table['D_CC'])/(triangle_table['D_R1']+triangle_table['D_R2']+2*triangle_table['D_CC'])

triangle_table.to_parquet(ot.Join(datafolder,'Triangle_Table.parquet'))
#%% ############## 2.Estimate Selection Index ####################

import matplotlib.ticker as ticker

# Define desired order for x-axis
network_order = ['MSB', 'ML', 'ASB', 'AL', 'Alexnet', 'VGG16', 'Res50']

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Palette for beautiful color, add personal favorite here
palette = "Set2"

box = sns.boxplot(
    data=triangle_table,
    x='Network',
    y='Selective_Index',
    hue='Layer',
    ax=ax,
    showfliers=False,
    width=0.9,
    order=network_order,
    palette=palette,
    boxprops=dict(alpha=.75, linewidth=1.5),
    whiskerprops=dict(linewidth=1.2),
    medianprops=dict(linewidth=2, color='k'),
    capprops=dict(linewidth=1.2)
)

# Remove top/right spines for a cleaner look
sns.despine(ax=ax)

# Add grid on y-axis for readability
ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)

# Add gray dashed horizontal line at y=0
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

# Set fonts and title for aesthetics
ax.set_title('Selection Index Across Networks', fontsize=18, fontweight='semibold', pad=18)
ax.set_xlabel('Network', fontsize=15)
ax.set_ylabel('Selective Index', fontsize=15)

# Improve x labels
ax.set_xticklabels(network_order, fontsize=13)

# Improve y ticks
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=8))
ax.tick_params(axis='y', labelsize=12)

# Move legend out for clarity
ax.legend(title='Layer', bbox_to_anchor=(1, 1), loc='upper left', frameon=False, fontsize=12, title_fontsize=13)

ax.set_yticks(np.round(np.arange(-0.2,1,0.2),2))
ax.set_yticklabels(np.round(np.arange(-0.2,1,0.2),2),fontsize=12)

fig.tight_layout(pad=2)
plt.show()


#%% ############### 3. Plot reversed triangles (one per Net-Layer) ####################
# Reversed (upside-down) triangle definition (isosceles):
# - base midpoint fixed at (0,0)
# - base length = mean(D_CC)   (edge: 4-4)
# - leg length  = mean((D_R1 + D_R2)/2)  (edges: 4-0)
# - upside down: tip at (0,0), base upward at y=height
# - Normalize so that (base + 2*leg) = 1 for each triangle

# --- Data Preparation ---
tri_plot = triangle_table.copy()
for c in ['D_R1', 'D_R2', 'D_CC']:
    tri_plot[c] = pd.to_numeric(tri_plot[c], errors='coerce')
tri_plot['D_leg'] = (tri_plot['D_R1'] + tri_plot['D_R2']) / 2

tri_agg = (
    tri_plot
    .groupby(['Network', 'Layer'], dropna=False, as_index=False)
    .agg(
        D_CC=('D_CC', 'mean'),
        D_leg=('D_leg', 'mean'),
        N=('D_CC', 'size')
    )
    .reset_index(drop=True)
)

# Normalize so that D_CC + 2*D_leg = 1 for each triangle
eps = 1e-9
tri_agg['norm_factor'] = tri_agg['D_CC'] + 2*tri_agg['D_leg'] + eps
tri_agg['D_CC_norm'] = tri_agg['D_CC'] / tri_agg['norm_factor']
tri_agg['D_leg_norm'] = tri_agg['D_leg'] / tri_agg['norm_factor']

# --- Custom Sorting (for hue order) ---
def net_layer_key(row):
    n, l = row['Network'], row['Layer']
    if n in ['MSB', 'ML', 'ASB', 'AL']:
        return ['MSB', 'ML', 'ASB', 'AL'].index(n)
    elif n == 'Alexnet' and l == 'last_conv':
        return 4
    elif n == 'VGG16' and l == 'last_conv':
        return 5
    elif n == 'Res50' and l == 'last_conv':
        return 6
    elif n == 'Alexnet' and l == 'fc6':
        return 7
    elif n == 'Alexnet' and l == 'fc7':
        return 8
    elif n == 'VGG16' and l == 'fc1':
        return 9
    elif n == 'VGG16' and l == 'fc2':
        return 10
    elif n == 'Res50' and l == 'avgpool':
        return 11
    else:
        return 99

tri_agg['hue_order'] = tri_agg.apply(net_layer_key, axis=1)
tri_agg = tri_agg.sort_values('hue_order').reset_index(drop=True)

# --- Color Assignment ---
from matplotlib.colors import to_hex
bio_networks = ['MSB', 'ML', 'ASB', 'AL']
conv_keys = []
fc_keys = []
for _, row in tri_agg.iterrows():
    n, l = row['Network'], row['Layer']
    key = f"{n}-{l}"
    if n in bio_networks:
        continue
    elif l == 'last_conv':
        conv_keys.append(key)
    elif l in ['fc6', 'fc7', 'fc1', 'fc2', 'avgpool']:
        fc_keys.append(key)

bio_color_list = sns.color_palette("Greens_r", n_colors=len(bio_networks))
conv_color_list = sns.color_palette("Reds_r", n_colors=len(conv_keys))
fc_color_list  = sns.color_palette("Blues_r", n_colors=len(fc_keys))

color_map = {}
for idx, n in enumerate(bio_networks):
    color_map.update({(n, l): to_hex(bio_color_list[idx]) for l in tri_agg[tri_agg['Network']==n]['Layer'].unique()})
for idx, key in enumerate(conv_keys):
    n, l = key.split('-', 1)
    color_map[(n, l)] = to_hex(conv_color_list[idx])
for idx, key in enumerate(fc_keys):
    n, l = key.split('-', 1)
    color_map[(n, l)] = to_hex(fc_color_list[idx])

# --- Fixed: triangle_vertices_upside_down ---
def triangle_vertices_upside_down(base, leg):
    """
    Given normalized base and leg lengths, return (x, y) vertices to plot an upside-down isosceles triangle.
    - base: normalized base length (width at the top, symmetric about x=0), placed at y=height
    - leg: normalized leg length (length from tip up to base ends)
    Returns:
        x, y (arrays of length 4, for a closed triangle: tip, base left, base right, back to tip)
    """
    # Height of triangle from tip upward (isosceles): h = sqrt(leg^2 - (base/2)^2)
    half_base = base / 2.0
    try:
        vertical = np.sqrt(np.clip(leg ** 2 - half_base ** 2, 0, None))
    except Exception:
        return None

    # Tip at (0,0) (bottom of triangle), base at y=vertical
    x = np.array([0, -half_base, half_base, 0])
    y = np.array([0,  vertical,  vertical, 0])
    return x, y

# --- Plot Triangles ---

xlim = (-0.25, 0.25)
ylim = (-0.05, 0.45)
figsize = (7.5, 7.5)

fig, ax = plt.subplots(1, 1, dpi=240, figsize=figsize)

for i, row in tri_agg.iterrows():
    verts = triangle_vertices_upside_down(row['D_CC_norm'], row['D_leg_norm'])
    if verts is None:
        continue
    x, y = verts
    network, layer = row['Network'], row['Layer']
    color = color_map.get((network, layer), 'gray')
    ax.plot(x, y, color=color, lw=1.7, alpha=0.95, label=f"{network}-{layer}")

ax.axhline(0, color='k', lw=0.8, alpha=0.25)
ax.axvline(0, color='k', lw=0.8, alpha=0.25)
ax.set_aspect('equal', adjustable='box')

if xlim is not None:
    ax.set_xlim(*xlim)
if ylim is not None:
    ax.set_ylim(*ylim)

# Remove duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
seen = set()
unique = []
for h, l in zip(handles, labels):
    if l not in seen:
        unique.append((h, l))
        seen.add(l)
if unique:
    h, l = zip(*unique)
    ax.legend(h, l, frameon=False, fontsize=7, loc='best')
else:
    ax.legend(frameon=False, fontsize=7, loc='best')

plt.show()




# %%
# Plot only msb, ml, asb, al, and all DCNNs in last_conv layer(s) using "raw" (not normalized) values
# Accepts any DCNN in tri_agg for last_conv only, and biological networks as usual

# List of biological network names
bio_networks = ['MSB', 'ML', 'ASB', 'AL']

# Define FC layers of interest (including fc7/fc2 for Alexnet/VGG16)
fc_layers = ['fc6', 'fc7', 'fc1', 'fc2', 'avgpool']

# Identify all DCNNs in tri_agg with last_conv or any fc layer
dcnn_lastconv = tri_agg[
    (~tri_agg['Network'].isin(bio_networks)) &
    (tri_agg['Layer'] == 'last_conv')
]['Network'].unique().tolist()

dcnn_fc = tri_agg[
    (~tri_agg['Network'].isin(bio_networks)) &
    (tri_agg['Layer'].isin(fc_layers))
]['Network'].unique().tolist()

# Compose full filter: biological (any layer), DCNNs last_conv, DCNNs fc/avgpool
tri_keep = tri_agg[
    (
        (tri_agg['Network'].isin(bio_networks)) |
        ((tri_agg['Network'].isin(dcnn_lastconv)) & (tri_agg['Layer'] == 'last_conv')) |
        ((tri_agg['Network'].isin(dcnn_fc)) & (tri_agg['Layer'].isin(fc_layers)))
    )
].reset_index(drop=True)

# Build ordered hue list: group by (bio, last_conv, fc), and keep track of networks
hue_order = []
hue_to_group = {}  # Map from hue_key to group (bio/lastconv/fc)
hue_to_network = {}  # Map from hue_key to network name

for n in bio_networks:
    bio_layers = tri_keep[(tri_keep['Network'] == n)]['Layer'].unique()
    for l in bio_layers:
        key = f"{n}-{l}"
        hue_order.append(key)
        hue_to_group[key] = 'bio'
        hue_to_network[key] = n

# Next all DCNNs in last_conv (order as they appear)
for n in dcnn_lastconv:
    if f"{n}-last_conv" in [f"{row['Network']}-{row['Layer']}" for _, row in tri_keep.iterrows()]:
        key = f"{n}-last_conv"
        hue_order.append(key)
        hue_to_group[key] = 'lastconv'
        hue_to_network[key] = n

# Next all DCNNs in fc layers -- now including fc7, fc2
for n in dcnn_fc:
    for l in fc_layers:
        if ((tri_keep['Network'] == n) & (tri_keep['Layer'] == l)).any():
            key = f"{n}-{l}"
            hue_order.append(key)
            hue_to_group[key] = 'fc'
            hue_to_network[key] = n

if tri_keep.empty:
    print("Warning: tri_keep DataFrame is empty. Check the Network names in tri_agg!")
else:
    print(f"tri_keep shape: {tri_keep.shape}")
    print("tri_keep Networks:", tri_keep['Network'].unique())
    print("D_CC min/max:", tri_keep['D_CC'].min(), tri_keep['D_CC'].max())
    print("D_leg min/max:", tri_keep['D_leg'].min(), tri_keep['D_leg'].max())

    # Filter for valid triangles
    valid_keep = tri_keep[np.isfinite(tri_keep['D_CC']) & np.isfinite(tri_keep['D_leg'])].copy()

    if valid_keep.empty:
        print("Warning: No valid (finite) D_CC or D_leg values to plot.")
    else:
        # Prepare hue keys and order as before
        valid_keep['hue_key'] = valid_keep['Network'] + '-' + valid_keep['Layer'].astype(str)
        order_mask = valid_keep['hue_key'].apply(lambda k: hue_order.index(k) if k in hue_order else 999)
        valid_keep = valid_keep.iloc[order_mask.argsort()].reset_index(drop=True)

        # Find distinct networks per group for color assignment
        bio_networks_present = sorted(list(valid_keep[valid_keep['Network'].isin(bio_networks)]['Network'].unique()))
        conv_networks = sorted(list(valid_keep[
            (~valid_keep['Network'].isin(bio_networks)) &
            (valid_keep['Layer'] == 'last_conv')
        ]['Network'].unique()))
        fc_networks = sorted(list(valid_keep[
            (~valid_keep['Network'].isin(bio_networks)) &
            (valid_keep['Layer'].isin(fc_layers))
        ]['Network'].unique()))

        # Build a color_map as in the cell above (the main triangle plot)
        # Set up a dict: (network, layer) -> color
        color_map = {}

        # Standard bright colors for matching the "cell above"
        import matplotlib.colors as mcolors

        group_base_colors = {
            'bio': '#33a02c',      # green (as above, formerly 'green')
            'lastconv': '#e31a1c', # red
            'fc': '#1f78b4',       # blue
        }

        def shades(base_hex, names):
            # Returns evenly lightened shades from base color for each given name
            if not names: return {}
            base = mcolors.to_rgb(base_hex)
            white = np.array([1.0,1.0,1.0])
            n = len(names)
            colors = []
            for i, name in enumerate(sorted(names)):
                alpha = (i/(n*1.85))  # Not too washed out
                color = tuple(base[j] + (white[j] - base[j])*alpha for j in range(3))
                colors.append(mcolors.to_hex(color))
            return dict(zip(sorted(names), colors))

        # Build shades for present networks in each group
        bio_shades = shades(group_base_colors['bio'], bio_networks_present)
        lastconv_shades = shades(group_base_colors['lastconv'], conv_networks)
        fc_shades = shades(group_base_colors['fc'], fc_networks)

        # Map to (network, layer)
        for k in hue_order:
            net = hue_to_network[k]
            group = hue_to_group[k]
            if group == 'bio':
                color = bio_shades.get(net, group_base_colors['bio'])
            elif group == 'lastconv':
                color = lastconv_shades.get(net, group_base_colors['lastconv'])
            elif group == 'fc':
                color = fc_shades.get(net, group_base_colors['fc'])
            else:
                color = 'gray'
            # k is "{net}-{layer}"
            color_map[(net, k.split('-',1)[1])] = color

        fig2, ax2 = plt.subplots(1, 1, dpi=240, figsize=figsize)
        n_plotted = 0
        for i, row in valid_keep.iterrows():
            xy = triangle_vertices_upside_down(row['D_CC'], row['D_leg'])
            if xy is None:
                continue
            x, y = xy
            network, layer = row['Network'], row['Layer']
            color = color_map.get((network, layer), 'gray')
            label = f"{network}-{layer} "
            ax2.plot(x, y, color=color, lw=1.7, alpha=0.95, label=label)
            n_plotted += 1

        if n_plotted == 0:
            print("Warning: No triangles were actually plotted.")
        else:
            ax2.axhline(0, color='k', lw=0.8, alpha=0.25)
            ax2.axvline(0, color='k', lw=0.8, alpha=0.25)
            ax2.set_aspect('equal', adjustable='box')
            # Defensive: Handle axes limits only if they are finite and distinct
            handles, labels = ax2.get_legend_handles_labels()
            seen = set()
            ordered = []
            for hue in hue_order:
                label = f"{hue} "
                if label in labels and label not in seen:
                    idx = labels.index(label)
                    ordered.append((handles[idx], labels[idx]))
                    seen.add(label)
            if ordered:
                h, l = zip(*ordered)
                ax2.legend(h, l, frameon=False, fontsize=7, loc='best')
            else:
                ax2.legend(frameon=False, fontsize=7, loc='best')
            ax2.set_xlim(-0.5, 0.5)
            ax2.set_ylim(-0.05, 0.95)
            plt.show()

            
# %%
