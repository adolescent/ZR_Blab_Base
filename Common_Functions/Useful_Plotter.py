'''

Some plotters for good 

'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import kaleido
import plotly.graph_objects as go


#%%
'''
Func 1, triangle Location map, for tuning observation
'''
# 1. 首先归一化数据
def Triangle_Normalize(data):
    """为三元图准备normalization"""
    # data = data.clip(0) # clip will ruin the data.
    data_values = data.values if isinstance(data, pd.DataFrame) else data
    
    # 确保所有值为正
    min_val = data_values.min()
    if min_val < 0:
        shifted = data_values - min_val + 0.001
    else:
        shifted = data_values + 0.001
    
    # 归一化到和为1
    row_sums = shifted.sum(axis=1, keepdims=True)
    return shifted / row_sums

def Triangle_FOB(input_frame,show=True,width=800,height=600,label=False):

    # generate ternary plot for 
    columns = input_frame.columns
    normalized_data = Triangle_Normalize(input_frame)
    # 2. 创建三元图
    fig = go.Figure(go.Scatterternary({
        
        'mode': 'markers',
        'a': normalized_data[:, 2],  # 对FOB，一般把Object放到顶点
        'b': normalized_data[:, 0],  # 对应分量
        'c': normalized_data[:, 1],  # 对应分量
        'marker': {
            'symbol': 2,
            'color': 'red',
            'size': 2,
            'line': {'width': 0}
        }
    }))
    
    # 3. 设置三元图布局
    if label:
        fig.update_layout(
        ternary=dict(
                sum=1,
                aaxis=dict(title=columns[2], tickformat='.2f'),
                baxis=dict(title=columns[0], tickformat='.2f'),
                caxis=dict(title=columns[1], tickformat='.2f')
            ),
            # title=f'三元图 - {len(input_frame)}个点',
            width=width,
            height=height
        )
    else:
        fig.update_layout(
        ternary=dict(
                sum=1,
                aaxis=dict(title=columns[2], tickformat='.2f',showticklabels=False,ticks=''),
                baxis=dict(title=columns[0], tickformat='.2f',showticklabels=False,ticks=''),
                caxis=dict(title=columns[1], tickformat='.2f',showticklabels=False,ticks='')
            ),
            # title=f'三元图 - {len(input_frame)}个点',
            width=width,
            height=height
        )
    if show:
        fig.show()

    return fig

#%% Tuning Radarmap
'''
Different from triangle map, radar map will retain raw response of all FOB tuning, a cell can be good at both categories. This shows how this neuron response to all locs, FOB, FOBSF optional.
'''
def Radar_FOB(input_frame,show=True,normalize=False):
    pass




