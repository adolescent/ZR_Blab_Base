'''

This script will calculate firing rate variation of each cell for each graph.
For each cell, norm its' animate average response as 1.
Then plot each brain area, each constrain level.
generate a data frame,but 


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
savepath = r'E:\#Preprocessed_Data\260305_Report_Data\Firing_Rate_Slope'
brain_areas = ['AL','ASB','ML','MSB']


#%% # get stim info matrix.
stim_infos = pd.DataFrame(index = range(200),columns = ['Img_Index','Shuffle','Ani'])
stim_infos['Img_Index'] = np.tile(np.arange(40), 5)
stim_infos['Shuffle'] = np.repeat(np.arange(5), 40)
stim_infos['Ani'] = stim_infos['Img_Index']<20
#%% calculate each cell's response ratio of 
Firing_Rate_Matrix = pd.DataFrame(columns=['Brain_Area', 'Cell', 'Baseline', 'Shuffle', 'Img_Index', 'Ratio', 'Raw_FR', 'Ani'])
for i, brain_area in tqdm(enumerate(brain_areas)):
    c_filename = f'{brain_area}_Cells_Metamer_Only.npz'
    reader = np.load(ot.Join(datafolder, c_filename), allow_pickle=True)
    psth = reader['psth'][:, :, 160:320].sum(-1)
    del reader
    # psth: (n_cell, 1000) after summing time window
    # reshape to (n_cell, n_repeat(=5), n_condition(=200)) then average repeats
    psth_avr = psth.reshape((len(psth), 5, 200)).mean(1)  # (n_cell, 200)
    
    # Make masks
    ani_mask = stim_infos['Ani'].to_numpy(dtype=bool)  # (200,)
    shuffle_mask = stim_infos['Shuffle'].to_numpy() == 0  # (200,)
    baseline_mask = ani_mask & shuffle_mask  # Only animate_raw (Ani==True & Shuffle==0)
    
    # Baseline now only from animate_raw (Ani==True & Shuffle==0)
    baseline = psth_avr[:, baseline_mask].mean(1)  # (n_cell,)
    
    # ratio per condition; keep NaN for cells with 0 baseline
    ratio = np.full_like(psth_avr, np.nan, dtype=float)
    valid = baseline > 0
    ratio[valid, :] = psth_avr[valid, :] / baseline[valid, None]

    n_cell = psth_avr.shape[0]
    # make Cell id unique across brain areas
    cc = np.repeat(np.array([f'{brain_area}_{k}' for k in range(n_cell)], dtype=object), 200)
    cond = np.tile(np.arange(200, dtype=int), n_cell)
    add_df = pd.DataFrame({
        'Brain_Area': np.repeat(brain_area, n_cell * 200),
        'Cell': cc,
        'Baseline': np.repeat(baseline, 200),
        'Shuffle': stim_infos['Shuffle'].to_numpy()[cond].astype(int),
        'Img_Index': stim_infos['Img_Index'].to_numpy()[cond].astype(int),
        'Ratio': ratio.reshape(-1),
        'Raw_FR': psth_avr.reshape(-1),
        'Ani': stim_infos['Ani'].to_numpy()[cond].astype(bool),
    })
    Firing_Rate_Matrix = pd.concat([Firing_Rate_Matrix, add_df], ignore_index=True)
Firing_Rate_Matrix.to_parquet(ot.Join(savepath,'FR_Raw.parquet'))
#%%

# Define color variations for each area and hue
# HARD-CODED legend/plot order
legend_area_order = ['MSB', 'ML', 'ASB', 'AL']
area_colors_inani = {
    'AL': '#0072B2',         # blue
    'ASB': '#258fc7',        # lighter/different blue
    'ML': '#005b90',         # darker blue
    'MSB': '#33aadd',        # another blue variant
}
area_colors_ani = {
    'AL': '#D55E00',         # orange
    'ASB': '#d5742f',        # lighter/different orange
    'ML': '#b14500',         # darker orange
    'MSB': '#dd8033',        # another orange variant
}

fig, ax = plt.subplots(figsize=(9,6), dpi=240)

plot_handles = []
plot_labels = []

# Ani plots first, following legend_area_order
for area in legend_area_order:
    ani = True
    color = area_colors_ani[area]
    plot_df = Firing_Rate_Matrix[
        (Firing_Rate_Matrix['Brain_Area'] == area) & (Firing_Rate_Matrix['Ani'] == ani)
    ].groupby('Shuffle', as_index=False)['Ratio'].mean()
    legend_label = f"{area}_Ani"
    handle, = ax.plot(
        plot_df['Shuffle'], 
        plot_df['Ratio'], 
        label=legend_label, 
        color=color, 
        linewidth=2.2, 
        marker='o'
    )
    plot_handles.append(handle)
    plot_labels.append(legend_label)

# Inani plots next, following legend_area_order
for area in legend_area_order:
    ani = False
    color = area_colors_inani[area]
    plot_df = Firing_Rate_Matrix[
        (Firing_Rate_Matrix['Brain_Area'] == area) & (Firing_Rate_Matrix['Ani'] == ani)
    ].groupby('Shuffle', as_index=False)['Ratio'].mean()
    legend_label = f"{area}_Inani"
    handle, = ax.plot(
        plot_df['Shuffle'], 
        plot_df['Ratio'], 
        label=legend_label, 
        color=color, 
        linewidth=2.2, 
        marker='o'
    )
    plot_handles.append(handle)
    plot_labels.append(legend_label)

ax.set_title('Firing Rate Ratio Across Shuffles (per Area)', fontsize=16, pad=15, fontweight='bold')
ax.set_xlabel('Shuffle', fontsize=14)
ax.set_ylabel('FR Ratio (Stim/Baseline)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(
    plot_handles, plot_labels,
    title=None, fontsize=8, loc='best', frameon=False
)
ax.grid(visible=True, which='major', axis='y', linestyle='--', alpha=0.35)
ax.set_xticks([0,1,2,3,4])
plt.tight_layout()
