'''

Generate 2-way anova for img_index and overall.
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

datafolder=r'E:\#Preprocessed_Data\Selected_Cells'
savepath = r'E:\#Preprocessed_Data\260305_Report_Data\Site_ANOVAs'

# filename = r'Res50_Response.npz'
# filename = r'Alex_Response.npz'

brain_area = 'AL'
filename = fr'{brain_area}_Cells_Metamer_Only.npz'

data = np.load(ot.Join(datafolder,filename))
keys = list(data.keys())
n_img = 1000
avr_rsp = data['psth'][:,:,160:320].sum(-1)
del data

#%% ########################## 2-Way ANOVA########################
# shuffle level and img index, which contribute more to each cell? This part will do 2-way anova on shuffle level and img index, getting explianed var of each cell.
## step 1, generating conding level of level 1 and level2
# Generate Img_Index column: 1-40 repeated 25 times (total 1000)
img_indices = np.tile(np.arange(1, 41), 25)
# Generate Shuffle Level column: for each of 0-4, repeat 40 times; the whole 0-4 block is repeated 25 times
shuffle_levels = np.tile(np.repeat(np.arange(5), 40), 5)
# Ensure arrays are the correct length
assert len(img_indices) == 1000
assert len(shuffle_levels) == 1000
df_conditions = pd.DataFrame({
    'Img_Index': img_indices,
    'Shuffle_Level': shuffle_levels
})

# Group split: Group1 = img_index 1-20, Group2 = img_index 21-40 (keep overall = all 1-40)
mask_g1 = (img_indices >= 1) & (img_indices <= 20)
mask_g2 = (img_indices >= 21) & (img_indices <= 40)
df_conditions_g1 = df_conditions.loc[mask_g1].reset_index(drop=True)   # 500 trials
df_conditions_g2 = df_conditions.loc[mask_g2].reset_index(drop=True)   # 500 trials
avr_rsp_g1 = avr_rsp[:, mask_g1]   # (n_cells, 500)
avr_rsp_g2 = avr_rsp[:, mask_g2]   # (n_cells, 500)
#%% Then for each cell, we generate 2-way anova, getting anova table for each cell.
################## Overall: img_index 1-40 ##################
################## Regard shuffle level as category var #########
import statsmodels.api as sm
from statsmodels.formula.api import ols

ANOVA_Table_Category = pd.DataFrame(
    0.0,
    index=range(len(avr_rsp)),
    columns=[
        'Img_SS', 'Shuffle_SS', 'Interact_SS',
        'Img_MS', 'Shuffle_MS', 'Interact_MS',
        'Residual_SS', 'Residual_MS',
        'F_Img', 'F_Shuffle', 'F_Interact',
        'Explained_VAR_Img', 'Explained_VAR_Shuffle', 'Explained_VAR_Interact',
        'Explained_VAR_ALL'  # Add the new column here
    ]
)

for i in tqdm(range(len(avr_rsp))):
    cc_response = avr_rsp[i, :]
    df_cell = df_conditions.copy()
    df_cell['Response'] = cc_response

    # Treat both variables as categorical for 2-way ANOVA
    model = ols('Response ~ C(Img_Index) * C(Shuffle_Level)', data=df_cell).fit()

    from statsmodels.stats.anova import anova_lm
    anova_res = anova_lm(model, typ=2)

    # Extract rows based on categorical coding
    img_row = anova_res.loc['C(Img_Index)']
    shuf_row = anova_res.loc['C(Shuffle_Level)']
    inter_row = anova_res.loc['C(Img_Index):C(Shuffle_Level)']
    resid_row = anova_res.loc['Residual']

    # Sum of squares
    img_ss = img_row['sum_sq']
    shuf_ss = shuf_row['sum_sq']
    inter_ss = inter_row['sum_sq']
    resid_ss = resid_row['sum_sq']

    # Mean squares
    img_ms = img_ss / img_row['df'] if img_row['df'] != 0 else np.nan
    shuf_ms = shuf_ss / shuf_row['df'] if shuf_row['df'] != 0 else np.nan
    inter_ms = inter_ss / inter_row['df'] if inter_row['df'] != 0 else np.nan
    resid_ms = resid_ss / resid_row['df'] if resid_row['df'] != 0 else np.nan

    # F values
    F_img = img_row['F']
    F_shuf = shuf_row['F']
    F_inter = inter_row['F']

    # Explained variance (proportion of total SS)
    total_ss = anova_res['sum_sq'].sum()
    ev_img = img_ss / total_ss if total_ss != 0 else np.nan
    ev_shuf = shuf_ss / total_ss if total_ss != 0 else np.nan
    ev_inter = inter_ss / total_ss if total_ss != 0 else np.nan

    # Explained variance for the whole model (everything except the residual)
    explained_var_all = 1.0 - (resid_ss / total_ss) if total_ss != 0 else np.nan

    ANOVA_Table_Category.loc[i, :] = [
        img_ss, shuf_ss, inter_ss,
        img_ms, shuf_ms, inter_ms,
        resid_ss, resid_ms,
        F_img, F_shuf, F_inter,
        ev_img, ev_shuf, ev_inter,
        explained_var_all  # Add the calculated value
    ]

# %%
################ Overall: regard img_index as category, but shuffle_level as linear var. ###################

# Add p-value columns to the DataFrame
ANOVA_Table_Linear = pd.DataFrame(
    0.0,
    index=range(len(avr_rsp)),
    columns=[
        'Img_SS', 'Shuffle_SS', 'Interact_SS',
        'Img_MS', 'Shuffle_MS', 'Interact_MS',
        'Residual_SS', 'Residual_MS',
        'F_Img', 'F_Shuffle', 'F_Interact',
        'p_Img', 'p_Shuffle', 'p_Interact',   # Added p-value columns
        'Explained_VAR_Img', 'Explained_VAR_Shuffle', 'Explained_VAR_Interact',
        'Explained_VAR_ALL'
    ]
)

for i in tqdm(range(len(avr_rsp))):
    cc_response = avr_rsp[i, :]
    df_cell = df_conditions.copy()
    df_cell['Response'] = cc_response

    # Treat Img_Index as categorical and Shuffle_Level as linear (numeric) for ANOVA
    model = ols('Response ~ C(Img_Index) * Shuffle_Level', data=df_cell).fit()

    from statsmodels.stats.anova import anova_lm
    anova_res = anova_lm(model, typ=2)

    # Extract rows based on model coding
    img_row = anova_res.loc['C(Img_Index)']
    shuf_row = anova_res.loc['Shuffle_Level']
    inter_row = anova_res.loc['C(Img_Index):Shuffle_Level']
    resid_row = anova_res.loc['Residual']

    # Sum of squares
    img_ss = img_row['sum_sq']
    shuf_ss = shuf_row['sum_sq']
    inter_ss = inter_row['sum_sq']
    resid_ss = resid_row['sum_sq']

    # Mean squares
    img_ms = img_ss / img_row['df'] if img_row['df'] != 0 else np.nan
    shuf_ms = shuf_ss / shuf_row['df'] if shuf_row['df'] != 0 else np.nan
    inter_ms = inter_ss / inter_row['df'] if inter_row['df'] != 0 else np.nan
    resid_ms = resid_ss / resid_row['df'] if resid_row['df'] != 0 else np.nan

    # F values
    F_img = img_row['F']
    F_shuf = shuf_row['F']
    F_inter = inter_row['F']

    # p-values
    p_img = img_row['PR(>F)']
    p_shuf = shuf_row['PR(>F)']
    p_inter = inter_row['PR(>F)']

    # Explained variance (proportion of total SS)
    total_ss = anova_res['sum_sq'].sum()
    ev_img = img_ss / total_ss if total_ss != 0 else np.nan
    ev_shuf = shuf_ss / total_ss if total_ss != 0 else np.nan
    ev_inter = inter_ss / total_ss if total_ss != 0 else np.nan

    # Explained variance for the whole model (everything except the residual)
    explained_var_all = 1.0 - (resid_ss / total_ss) if total_ss != 0 else np.nan

    ANOVA_Table_Linear.loc[i, :] = [
        img_ss, shuf_ss, inter_ss,
        img_ms, shuf_ms, inter_ms,
        resid_ss, resid_ms,
        F_img, F_shuf, F_inter,
        p_img, p_shuf, p_inter,    # Fill new columns
        ev_img, ev_shuf, ev_inter,
        explained_var_all
    ]

#%%
################ Group1 (img_index 1-20): categorical shuffle ##################
ANOVA_Table_Category_Group1 = pd.DataFrame(
    0.0,
    index=range(len(avr_rsp)),
    columns=[
        'Img_SS', 'Shuffle_SS', 'Interact_SS',
        'Img_MS', 'Shuffle_MS', 'Interact_MS',
        'Residual_SS', 'Residual_MS',
        'F_Img', 'F_Shuffle', 'F_Interact',
        'Explained_VAR_Img', 'Explained_VAR_Shuffle', 'Explained_VAR_Interact',
        'Explained_VAR_ALL'
    ]
)
for i in tqdm(range(len(avr_rsp)), desc='ANOVA Cat Group1'):
    df_cell = df_conditions_g1.copy()
    df_cell['Response'] = avr_rsp_g1[i, :]
    model = ols('Response ~ C(Img_Index) * C(Shuffle_Level)', data=df_cell).fit()
    anova_res = anova_lm(model, typ=2)
    img_row = anova_res.loc['C(Img_Index)']
    shuf_row = anova_res.loc['C(Shuffle_Level)']
    inter_row = anova_res.loc['C(Img_Index):C(Shuffle_Level)']
    resid_row = anova_res.loc['Residual']
    img_ss = img_row['sum_sq']; shuf_ss = shuf_row['sum_sq']; inter_ss = inter_row['sum_sq']; resid_ss = resid_row['sum_sq']
    img_ms = img_ss / img_row['df'] if img_row['df'] != 0 else np.nan
    shuf_ms = shuf_ss / shuf_row['df'] if shuf_row['df'] != 0 else np.nan
    inter_ms = inter_ss / inter_row['df'] if inter_row['df'] != 0 else np.nan
    resid_ms = resid_ss / resid_row['df'] if resid_row['df'] != 0 else np.nan
    total_ss = anova_res['sum_sq'].sum()
    ev_img = img_ss / total_ss if total_ss != 0 else np.nan
    ev_shuf = shuf_ss / total_ss if total_ss != 0 else np.nan
    ev_inter = inter_ss / total_ss if total_ss != 0 else np.nan
    explained_var_all = 1.0 - (resid_ss / total_ss) if total_ss != 0 else np.nan
    ANOVA_Table_Category_Group1.loc[i, :] = [
        img_ss, shuf_ss, inter_ss, img_ms, shuf_ms, inter_ms, resid_ss, resid_ms,
        img_row['F'], shuf_row['F'], inter_row['F'],
        ev_img, ev_shuf, ev_inter, explained_var_all
    ]

################ Group1 (img_index 1-20): linear shuffle ##################
ANOVA_Table_Linear_Group1 = pd.DataFrame(
    0.0,
    index=range(len(avr_rsp)),
    columns=[
        'Img_SS', 'Shuffle_SS', 'Interact_SS',
        'Img_MS', 'Shuffle_MS', 'Interact_MS',
        'Residual_SS', 'Residual_MS',
        'F_Img', 'F_Shuffle', 'F_Interact',
        'p_Img', 'p_Shuffle', 'p_Interact',
        'Explained_VAR_Img', 'Explained_VAR_Shuffle', 'Explained_VAR_Interact',
        'Explained_VAR_ALL'
    ]
)
for i in tqdm(range(len(avr_rsp)), desc='ANOVA Linear Group1'):
    df_cell = df_conditions_g1.copy()
    df_cell['Response'] = avr_rsp_g1[i, :]
    model = ols('Response ~ C(Img_Index) * Shuffle_Level', data=df_cell).fit()
    anova_res = anova_lm(model, typ=2)
    img_row = anova_res.loc['C(Img_Index)']; shuf_row = anova_res.loc['Shuffle_Level']; inter_row = anova_res.loc['C(Img_Index):Shuffle_Level']; resid_row = anova_res.loc['Residual']
    img_ss = img_row['sum_sq']; shuf_ss = shuf_row['sum_sq']; inter_ss = inter_row['sum_sq']; resid_ss = resid_row['sum_sq']
    img_ms = img_ss / img_row['df'] if img_row['df'] != 0 else np.nan
    shuf_ms = shuf_ss / shuf_row['df'] if shuf_row['df'] != 0 else np.nan
    inter_ms = inter_ss / inter_row['df'] if inter_row['df'] != 0 else np.nan
    resid_ms = resid_ss / resid_row['df'] if resid_row['df'] != 0 else np.nan
    total_ss = anova_res['sum_sq'].sum()
    ev_img = img_ss / total_ss if total_ss != 0 else np.nan
    ev_shuf = shuf_ss / total_ss if total_ss != 0 else np.nan
    ev_inter = inter_ss / total_ss if total_ss != 0 else np.nan
    explained_var_all = 1.0 - (resid_ss / total_ss) if total_ss != 0 else np.nan
    ANOVA_Table_Linear_Group1.loc[i, :] = [
        img_ss, shuf_ss, inter_ss, img_ms, shuf_ms, inter_ms, resid_ss, resid_ms,
        img_row['F'], shuf_row['F'], inter_row['F'],
        img_row['PR(>F)'], shuf_row['PR(>F)'], inter_row['PR(>F)'],
        ev_img, ev_shuf, ev_inter, explained_var_all
    ]

#%%
################ Group2 (img_index 21-40): categorical shuffle ##################
ANOVA_Table_Category_Group2 = pd.DataFrame(
    0.0,
    index=range(len(avr_rsp)),
    columns=[
        'Img_SS', 'Shuffle_SS', 'Interact_SS',
        'Img_MS', 'Shuffle_MS', 'Interact_MS',
        'Residual_SS', 'Residual_MS',
        'F_Img', 'F_Shuffle', 'F_Interact',
        'Explained_VAR_Img', 'Explained_VAR_Shuffle', 'Explained_VAR_Interact',
        'Explained_VAR_ALL'
    ]
)
for i in tqdm(range(len(avr_rsp)), desc='ANOVA Cat Group2'):
    df_cell = df_conditions_g2.copy()
    df_cell['Response'] = avr_rsp_g2[i, :]
    model = ols('Response ~ C(Img_Index) * C(Shuffle_Level)', data=df_cell).fit()
    anova_res = anova_lm(model, typ=2)
    img_row = anova_res.loc['C(Img_Index)']
    shuf_row = anova_res.loc['C(Shuffle_Level)']
    inter_row = anova_res.loc['C(Img_Index):C(Shuffle_Level)']
    resid_row = anova_res.loc['Residual']
    img_ss = img_row['sum_sq']; shuf_ss = shuf_row['sum_sq']; inter_ss = inter_row['sum_sq']; resid_ss = resid_row['sum_sq']
    img_ms = img_ss / img_row['df'] if img_row['df'] != 0 else np.nan
    shuf_ms = shuf_ss / shuf_row['df'] if shuf_row['df'] != 0 else np.nan
    inter_ms = inter_ss / inter_row['df'] if inter_row['df'] != 0 else np.nan
    resid_ms = resid_ss / resid_row['df'] if resid_row['df'] != 0 else np.nan
    total_ss = anova_res['sum_sq'].sum()
    ev_img = img_ss / total_ss if total_ss != 0 else np.nan
    ev_shuf = shuf_ss / total_ss if total_ss != 0 else np.nan
    ev_inter = inter_ss / total_ss if total_ss != 0 else np.nan
    explained_var_all = 1.0 - (resid_ss / total_ss) if total_ss != 0 else np.nan
    ANOVA_Table_Category_Group2.loc[i, :] = [
        img_ss, shuf_ss, inter_ss, img_ms, shuf_ms, inter_ms, resid_ss, resid_ms,
        img_row['F'], shuf_row['F'], inter_row['F'],
        ev_img, ev_shuf, ev_inter, explained_var_all
    ]

################ Group2 (img_index 21-40): linear shuffle ##################
ANOVA_Table_Linear_Group2 = pd.DataFrame(
    0.0,
    index=range(len(avr_rsp)),
    columns=[
        'Img_SS', 'Shuffle_SS', 'Interact_SS',
        'Img_MS', 'Shuffle_MS', 'Interact_MS',
        'Residual_SS', 'Residual_MS',
        'F_Img', 'F_Shuffle', 'F_Interact',
        'p_Img', 'p_Shuffle', 'p_Interact',
        'Explained_VAR_Img', 'Explained_VAR_Shuffle', 'Explained_VAR_Interact',
        'Explained_VAR_ALL'
    ]
)
for i in tqdm(range(len(avr_rsp)), desc='ANOVA Linear Group2'):
    df_cell = df_conditions_g2.copy()
    df_cell['Response'] = avr_rsp_g2[i, :]
    model = ols('Response ~ C(Img_Index) * Shuffle_Level', data=df_cell).fit()
    anova_res = anova_lm(model, typ=2)
    img_row = anova_res.loc['C(Img_Index)']; shuf_row = anova_res.loc['Shuffle_Level']; inter_row = anova_res.loc['C(Img_Index):Shuffle_Level']; resid_row = anova_res.loc['Residual']
    img_ss = img_row['sum_sq']; shuf_ss = shuf_row['sum_sq']; inter_ss = inter_row['sum_sq']; resid_ss = resid_row['sum_sq']
    img_ms = img_ss / img_row['df'] if img_row['df'] != 0 else np.nan
    shuf_ms = shuf_ss / shuf_row['df'] if shuf_row['df'] != 0 else np.nan
    inter_ms = inter_ss / inter_row['df'] if inter_row['df'] != 0 else np.nan
    resid_ms = resid_ss / resid_row['df'] if resid_row['df'] != 0 else np.nan
    total_ss = anova_res['sum_sq'].sum()
    ev_img = img_ss / total_ss if total_ss != 0 else np.nan
    ev_shuf = shuf_ss / total_ss if total_ss != 0 else np.nan
    ev_inter = inter_ss / total_ss if total_ss != 0 else np.nan
    explained_var_all = 1.0 - (resid_ss / total_ss) if total_ss != 0 else np.nan
    ANOVA_Table_Linear_Group2.loc[i, :] = [
        img_ss, shuf_ss, inter_ss, img_ms, shuf_ms, inter_ms, resid_ss, resid_ms,
        img_row['F'], shuf_row['F'], inter_row['F'],
        img_row['PR(>F)'], shuf_row['PR(>F)'], inter_row['PR(>F)'],
        ev_img, ev_shuf, ev_inter, explained_var_all
    ]
#%% Plot demo of explained variance
fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=160)
bins = np.linspace(0, 1, 25)

# Explained_VAR_ALL
axes[0].hist(ANOVA_Table_Linear_Group1['Explained_VAR_ALL'], bins=bins, alpha=0.7, label='Group 1', color='C0')
axes[0].hist(ANOVA_Table_Linear_Group2['Explained_VAR_ALL'], bins=bins, alpha=0.7, label='Group 2', color='C1')
axes[0].hist(ANOVA_Table_Linear['Explained_VAR_ALL'], bins=bins, alpha=0.5, label='All', color='gray')
axes[0].set_title('Explained_VAR_ALL')
axes[0].set_xlabel('Explained Variance')
axes[0].set_ylabel('Cell count')
axes[0].legend()

# Explained_VAR_Shuffle
axes[1].hist(ANOVA_Table_Linear_Group1['Explained_VAR_Shuffle'], bins=bins, alpha=0.7, label='Group 1', color='C0')
axes[1].hist(ANOVA_Table_Linear_Group2['Explained_VAR_Shuffle'], bins=bins, alpha=0.7, label='Group 2', color='C1')
axes[1].hist(ANOVA_Table_Linear['Explained_VAR_Shuffle'], bins=bins, alpha=0.5, label='All', color='gray')
axes[1].set_title('Explained_VAR_Shuffle')
axes[1].set_xlabel('Explained Variance')
axes[1].set_ylabel('Cell count')
axes[1].legend()

# Overlayed Combined view (both metrics as lines)
axes[2].hist(ANOVA_Table_Linear_Group1['Explained_VAR_ALL'], bins=bins, alpha=0.5, label='All (Grp1)', color='C0', histtype='step')
axes[2].hist(ANOVA_Table_Linear_Group1['Explained_VAR_Shuffle'], bins=bins, alpha=0.5, label='Shuffle (Grp1)', color='C0', linestyle='--', histtype='step')
axes[2].hist(ANOVA_Table_Linear_Group2['Explained_VAR_ALL'], bins=bins, alpha=0.5, label='All (Grp2)', color='C1', histtype='step')
axes[2].hist(ANOVA_Table_Linear_Group2['Explained_VAR_Shuffle'], bins=bins, alpha=0.5, label='Shuffle (Grp2)', color='C1', linestyle='--', histtype='step')
axes[2].set_title('Overlay: ALL vs Shuffle (Grp1 & Grp2)')
axes[2].set_xlabel('Explained Variance')
axes[2].set_ylabel('Cell count')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()



#%% last part, save results.
ANOVA_Table_Linear_Group1.to_parquet(ot.Join(savepath,f'Ani_{brain_area}_ANOVA.parquet'))
ANOVA_Table_Linear_Group2.to_parquet(ot.Join(savepath,f'Inani_{brain_area}_ANOVA.parquet'))
ANOVA_Table_Linear.to_parquet(ot.Join(savepath,f'All_{brain_area}_ANOVA.parquet'))


#%%
################ Demo heatmaps: single cell and average over cells (normalized) ###################
# Layout: overall 1000 = 5 repeats × (5 shuffle × 40 img). Group1/Group2: 500 = 5 × (5 × 20).

N_Shuffle = 5
n_repeats = 5
N_img = 40

def response_to_heatmap_matrix(response_1d, n_img):
    """Reshape (n_repeats*N_Shuffle*n_img,) to (N_Shuffle, n_img) by averaging over repeats."""
    r = response_1d.reshape(n_repeats, N_Shuffle, n_img)
    return r.mean(axis=0)  # (N_Shuffle, n_img)

#%% --- Averaged heatmaps ---
N_img_g = 20
vmax = 2.6
vmin = 0
cmap='plasma'

# Group 1: images 1-20
c_rsp_g1 = avr_rsp_g1 / (avr_rsp_g1.std(axis=1, keepdims=True) + 1e-12)
c_rsp_g1 = np.clip(c_rsp_g1, 0, 10)
all_g1 = np.array([response_to_heatmap_matrix(c_rsp_g1[i, :], N_img_g) for i in range(len(avr_rsp))]).mean(axis=0)

# Group 2: images 21-40
c_rsp_g2 = avr_rsp_g2 / (avr_rsp_g2.std(axis=1, keepdims=True) + 1e-12)
c_rsp_g2 = np.clip(c_rsp_g2, 0, 10)
all_g2 = np.array([response_to_heatmap_matrix(c_rsp_g2[i, :], N_img_g) for i in range(len(avr_rsp))]).mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Create a common colorbar axis right of both subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.1)

# Draw both heatmaps, but only use colorbar for the second one (so one cbar for both)
sns.heatmap(
    all_g1, ax=axes[0], 
    xticklabels=np.arange(1, N_img_g + 1), yticklabels=np.arange(N_Shuffle),
    cbar=False, vmin=vmin, vmax=vmax,cmap=cmap)
axes[0].set_title(f'{brain_area} Cells avg, images 1-20')
axes[0].set_xlabel('Img index')
axes[0].set_ylabel('Shuffle level')

sns.heatmap(
    all_g2, ax=axes[1], 
    xticklabels=np.arange(21, 21+N_img_g), yticklabels=np.arange(N_Shuffle),
    cbar=True, cbar_ax=cax, vmin=vmin, vmax=vmax,cmap=cmap,
    cbar_kws={'label': 'Response'}
)
axes[1].set_title(f'{brain_area} Cells avg, images 21-40')
axes[1].set_xlabel('Img index')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()


# %%
# --- Heatmap for a single cell, user selects cell ID ---

# Cell ID to visualize (index, 0-based); update as desired
cell_id = 880  # e.g., 0 for first cell; change to any in range(len(avr_rsp))

# Group 1: images 1-20 for single cell
# c_rsp_g1_single = avr_rsp_g1[cell_id] / (avr_rsp_g1[cell_id].std() + 1e-12)
# c_rsp_g1_single = np.clip(c_rsp_g1_single, 0, 10)
c_rsp_g1_single = avr_rsp_g1[cell_id] 
heatmap_g1 = response_to_heatmap_matrix(c_rsp_g1_single, N_img_g)

# Group 2: images 21-40 for single cell
# c_rsp_g2_single = avr_rsp_g2[cell_id] / (avr_rsp_g2[cell_id].std() + 1e-12)
# c_rsp_g2_single = np.clip(c_rsp_g2_single, 0, 10)
c_rsp_g2_single = avr_rsp_g2[cell_id] 
heatmap_g2 = response_to_heatmap_matrix(c_rsp_g2_single, N_img_g)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.1)

sns.heatmap(
    heatmap_g1, ax=axes[0],
    xticklabels=np.arange(1, N_img_g + 1), yticklabels=np.arange(N_Shuffle),
    cbar=False,  cmap=cmap
)
axes[0].set_title(f'Cell {cell_id}, images 1-20')
axes[0].set_xlabel('Img index')
axes[0].set_ylabel('Shuffle level')

sns.heatmap(
    heatmap_g2, ax=axes[1],
    xticklabels=np.arange(21, 21+N_img_g), yticklabels=np.arange(N_Shuffle),
    cbar=True, cbar_ax=cax,  cmap=cmap,
    cbar_kws={'label': 'Response'}
)
axes[1].set_title(f'Cell {cell_id}, images 21-40')
axes[1].set_xlabel('Img index')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
# %%
