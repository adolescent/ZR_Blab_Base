'''
比较DCNN的rsa以及平均图和真实神经数据的关系


'''

#%% Load in all rsa and corr 

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
# filename = r'Res50_Response.npz'
# filename = r'Alex_Response.npz'
al_name = r'AL_Cells_Metamer_Only.npz'
asb_name = r'ASB_Cells_Metamer_Only.npz'
ml_name = r'ML_Cells_Metamer_Only.npz'
msb_name = r'MSB_Cells_Metamer_Only.npz'
alex_path = r'Alex_Response.npz'
vgg_path = r'VGG16_Response.npz'
res_path = r'Res50_Response.npz'


#%% ################## Plot 1, Response_RSA_Similarity  ####################
# load rsa of all networks.
RSA_Dict_Neuron = {}
RSA_Dict_DCNN = {}

# read in neuron rsa.
neuro_sites = ['AL','ASB','ML','MSB']
neuro_file = [al_name,asb_name,ml_name,msb_name]
for i,c_site in tqdm(enumerate(neuro_file)):
    data = np.load(ot.Join(datafolder,c_site),allow_pickle=True)['psth'][:,:,160:320].sum(-1)
    c_response = (data)/data.std(1,keepdims = True)
    c_response = np.clip(c_response,0,10)
    c_rsa = Corr_Matrix(c_response,fill_diag=False)
    RSA_Dict_Neuron[neuro_sites[i]]=c_rsa

# read in dcnn response.
dcnn_sites = ['Alexnet','VGG16','Res50']
dcnn_file = [alex_path,vgg_path,res_path]
for i,c_site in tqdm(enumerate(dcnn_file)):
    c_net = dcnn_sites[i]
    data = np.load(ot.Join(datafolder,c_site),allow_pickle=True)
    if c_net == 'Alexnet':
        c_fc = data['fc6']
        c_conv = data['last_conv'].reshape(1000,-1)
    elif c_net == 'VGG16':
        c_fc = data['fc1']
        c_conv = data['last_conv'].reshape(1000,-1)
    elif c_net == 'Res50':
        c_fc = data['avgpool']
        c_conv = data['last_conv'].reshape(1000,-1)

    conv_rsa = Corr_Matrix(c_conv.T,fill_diag=False)
    fc_rsa = Corr_Matrix(c_fc.T,fill_diag=False)
    RSA_Dict_DCNN[c_net+'_fc'] = fc_rsa
    RSA_Dict_DCNN[c_net+'_last_conv'] = conv_rsa
#%% calculate corr between neuron net with all fc, and with all conv.
# correlation = np.corrcoef(matrix1.ravel(), matrix2.ravel())[0, 1]
indexes = ['MSB','ML','ASB','AL']
columns = ['Alexnet_last_conv','VGG16_last_conv','Res50_last_conv','Alexnet_fc','VGG16_fc','Res50_fc']
RSA_Corr = pd.DataFrame(0.0,columns = columns,index=indexes)

for i,c_site in enumerate(indexes):
    c_neuro = RSA_Dict_Neuron[c_site]
    for j,c_net in enumerate(columns):
        c_dcnn = RSA_Dict_DCNN[c_net]
        correlation = np.corrcoef(c_neuro.ravel(), c_dcnn.ravel())[0, 1]
        RSA_Corr.loc[c_site,c_net] = correlation

# plot similarity,not very good..
# sns.heatmap(RSA_Corr,center=0)

#%% ################## Plot 2, Constrain Averaged Similarity  ####################
datafolder=r'E:\#Preprocessed_Data\260305_Report_Data\Site_Constrain_Corr'
al_name = r'AL_Corr.parquet'
asb_name = r'ASB_Corr.parquet'
ml_name = r'ML_Corr.parquet'
msb_name = r'MSB_Corr.parquet'
alex_path = r'Alex_Corr.parquet'
vgg_path = r'VGG16_Corr.parquet'
res_path = r'Res50_Corr.parquet'

site_counter = 0
for i,c_path in enumerate([al_name,asb_name,ml_name,msb_name,alex_path,vgg_path,res_path]):
    c_frame = pd.read_parquet(ot.Join(datafolder,c_path))
    if site_counter == 0:
        Corr_Constrain = copy.deepcopy(c_frame)
    else:
        Corr_Constrain = pd.concat([Corr_Constrain,c_frame])
    site_counter += 1 

Corr_Constrain.to_parquet(ot.Join(datafolder,'Constrain_Corr.parquet'))
#%% get avr 
# 1. 确保镜像对称
plotable = Corr_Constrain

df_mirror = plotable.copy()
# 假设你的列名是 C_img1, C_img2, Corr
df_mirror.columns = ['Network','Layer', 'Img_Index', 'C_img2', 'C_img1','Corr','Dist']
df_total = pd.concat([plotable, df_mirror], axis=0)

df_total['C_img1'] = df_total['C_img1']%5
df_total['C_img2'] = df_total['C_img2']%5



# generate all corr matrix.
indexes = ['MSB','ML','ASB','AL']
columns = ['Alexnet','Res50','VGG16','Alexnet','Res50','VGG16']
layers = ['fc6','avgpool','fc1','last_conv','last_conv','last_conv']
real_columns =  [f"{c}_{l}" for c, l in zip(columns, layers)]
Combined_Corr = pd.DataFrame(0.0,columns = real_columns,index=indexes)
#%%
def get_avg_matrix(df, network, layer):
    """Return averaged 5x5 correlation matrix for given network & layer."""
    sub = df[(df['Network'] == network) & (df['Layer'] == layer)]
    mat = sub.pivot_table(
        index='C_img1',
        columns='C_img2',
        values='Corr',
        aggfunc='mean'
    )
    # ensure consistent ordering of indices
    mat = mat.sort_index().sort_index(axis=1)
    return mat.to_numpy()


# pre-compute averaged matrices for all neuron sites and DCNN layers
site_mats = {}
for site in indexes:
    site_mats[site] = get_avg_matrix(df_total, site, site)

dcnn_keys = list(zip(columns, layers, real_columns))
dcnn_mats = {}
for net, layer, col_name in dcnn_keys:
    dcnn_mats[col_name] = get_avg_matrix(df_total, net, layer)

# fill Combined_Corr with correlations between neuron sites and DCNN layers
for site in indexes:
    site_vec = site_mats[site].ravel()
    for col_name in real_columns:
        dcnn_vec = dcnn_mats[col_name].ravel()
        Combined_Corr.loc[site, col_name] = np.corrcoef(site_vec, dcnn_vec)[0, 1]
#%%
fig, ax = plt.subplots(ncols=1, nrows=1, dpi=240, figsize=(7, 5))
# Draw the heatmap (don't show yet)
heatmap = sns.heatmap(
    Combined_Corr,
    center=0.6, square=True,
    # vmin=0,
    vmax=1,
    cmap='RdBu_r',
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Pearson r'},
    ax=ax
)

# ---- Adjustable colorbar width and height ----

# Define adjustable width and height as fraction of figure (0-1)
cbar_width_frac = 0.01   # E.g., 0.03 for "normal", 0.05 for wider, up to ~0.2
cbar_height_frac =0.5  # Default: 1.0 (full height of ax). 0.33 = 1/3 height

# Draw to compute positions
fig.canvas.draw()

# Get colorbar and axes
cbar = heatmap.collections[0].colorbar
cbar_ax = cbar.ax
axpos = ax.get_position()
cbar_pos = cbar_ax.get_position()

# Set colorbar to be right of ax, and center height (absolute in figure coordinates)
new_cbar_x0 = axpos.x1 + 0.02  # 0.02 gap to the right of main ax
new_cbar_width = cbar_width_frac
new_cbar_height = axpos.height * cbar_height_frac
new_cbar_y0 = axpos.y0 + (axpos.height - new_cbar_height) / 2

cbar_ax.set_position([new_cbar_x0, new_cbar_y0, new_cbar_width, new_cbar_height])

# ---- End adjustable colorbar code ----

pretty_net = {'Alexnet': 'AlexNet', 'Res50': 'ResNet-50', 'VGG16': 'VGG16'}
pretty_layer = {'fc6': 'FC6', 'avgpool': 'AvgPool', 'fc1': 'FC1', 'last_conv': 'Last conv'}
xticklabels = []
for c in Combined_Corr.columns:
    net, layer = c.split('_', 1)
    xticklabels.append(f"{pretty_net.get(net, net)}\n{pretty_layer.get(layer, layer)}")

ax.set_xticklabels(xticklabels, rotation=0, ha='center', fontsize=9)
ax.set_yticklabels(list(Combined_Corr.index), rotation=0, fontsize=10)
ax.set_xlabel('DCNN layer', fontsize=11, labelpad=8)
ax.set_ylabel('Neuron site', fontsize=11, labelpad=8)
ax.tick_params(axis='both', length=0)
ax.set_title('Neuron–DCNN similarity (averaged constraint corr)', fontsize=12, pad=10)
plt.show()

