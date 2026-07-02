'''
做一些统计的分析，分析能否从中得到一些有趣的内容

'''


#%%

from pathlib import Path

import pandas as pd

fit_result_path = Path(
    r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Normalization_Bubble\Normalization_Results'
)
stats_savepath = Path(
    r'E:\#Preprocessed_Data\Selected_Cells\Metamers\Analysis\Normalization_Bubble\Stat_Results'
)
stats_savepath.mkdir(parents=True, exist_ok=True)

AREAS = ['ML', 'MSB', 'AL', 'ASB']


#%% 统计1 - 四个脑区的 bubble 拟合 R2 与 full-size image 预测 R2

fit_df = pd.read_pickle(fit_result_path / 'normalization_bubble_fit.pkl')
pred_cell_df = pd.read_pickle(fit_result_path / 'normalization_bubble_raw_pred_cell.pkl')

area_fit_r2_df = pd.DataFrame({
    'bubble_fit_r2': fit_df.groupby('area')['r2'].median(),
    'full_image_r2': pred_cell_df.groupby('area')['r2'].median(),
}).reindex(AREAS)

area_fit_r2_df

#%% boxplot - 各脑区 bubble 拟合 R2 分布（每个点 = 一个图片-神经元对）

import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(data=fit_df, x='area', y='r2', order=AREAS,showfliers=False)
plt.ylabel('bubble_fit_r2')
plt.xlabel('area')
plt.title('Normalization model fit R2 per image-neuron pair')
plt.tight_layout()
plt.show()

#%% 统计2 - 各脑区、各图片的激活野/抑制野大小（std）比较

K_MIN = 0.5   # 只保留 k > K_MIN 的拟合

good_fit = fit_df.loc[fit_df['k'] > K_MIN, ['area', 'object_id', 'active_std', 'negative_std']]

std_long = good_fit.melt(
    id_vars=['area', 'object_id'],
    value_vars=['active_std', 'negative_std'],
    var_name='field',
    value_name='std',
)
std_long['field'] = std_long['field'].map({'active_std': 'active', 'negative_std': 'negative'})

# 汇总表：每张图 × 脑区 × 野类型的 median std
std_summary_df = (
    std_long
    .groupby(['object_id', 'area', 'field'], as_index=False)['std']
    .median()
    .pivot(index=['object_id', 'area'], columns='field', values='std')
    .reset_index()
)

g = sns.catplot(
    data=std_long, x='area', y='std', hue='field', col='object_id',
    order=AREAS, hue_order=['active', 'negative'],
    kind='box', col_wrap=5, sharey=True, showfliers=False, height=2.5,
)
g.set_axis_labels('area', 'field std (px)')
g.set_titles('img {col_name}')
g.fig.suptitle(f'Active / negative field size (k > {K_MIN})', y=1.02)
plt.show()

std_summary_df

#%% 统计3 - 激活/抑制野中心距离 vs 随机配对

import numpy as np

K_MIN = 0.5
N_PERM = 200

center_df = fit_df.loc[
    fit_df['k'] > K_MIN,
    ['area', 'object_id', 'active_x', 'active_y', 'negative_x', 'negative_y'],
].copy()
center_df['center_dist'] = np.hypot(
    center_df['active_x'] - center_df['negative_x'],
    center_df['active_y'] - center_df['negative_y'],
)


def perm_null_median(area_df, rng):
    # 在同一张图内 shuffle 抑制野中心，打破与激活野的配对
    null_dists = []
    for _ in range(N_PERM):
        parts = []
        for _, g in area_df.groupby('object_id'):
            ax, ay = g['active_x'].to_numpy(), g['active_y'].to_numpy()
            nx, ny = g['negative_x'].to_numpy(), g['negative_y'].to_numpy()
            idx = rng.permutation(len(g))
            parts.append(np.hypot(ax - nx[idx], ay - ny[idx]))
        null_dists.append(np.median(np.concatenate(parts)))
    return np.array(null_dists)


rng = np.random.default_rng(42)
center_overlap_rows = []
for area in AREAS:
    g = center_df.loc[center_df['area'] == area]
    obs = g['center_dist'].median()
    null = perm_null_median(g, rng)
    center_overlap_rows.append({
        'area': area,
        'median_center_dist': obs,
        'null_median_dist': np.median(null),
        'p_closer_than_null': float((null <= obs).mean()),
        'p_farther_than_null': float((null >= obs).mean()),
    })

center_overlap_df = pd.DataFrame(center_overlap_rows).set_index('area').reindex(AREAS)

sns.boxplot(data=center_df, x='area', y='center_dist', order=AREAS, showfliers=False)
plt.ylabel('active-negative center distance (px)')
plt.title(f'Center overlap (k > {K_MIN})')
plt.tight_layout()
plt.show()

sns.barplot(
    data=center_overlap_df.reset_index().melt(
        id_vars='area', value_vars=['median_center_dist', 'null_median_dist'],
        var_name='type', value_name='dist',
    ),
    x='area', y='dist', hue='type', order=AREAS,
)
plt.ylabel('median center distance (px)')
plt.title('Observed vs permuted null')
plt.tight_layout()
plt.show()

center_overlap_df

#%% 统计4-展示激活野和抑制野中的图片特征，是否存在统计规律



