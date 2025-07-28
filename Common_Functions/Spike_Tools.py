'''
These functions are used for spike data processing.
'''


import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from Matrix_Tools import *
from scipy import stats
import OS_Tools as ot
import pandas as pd


def Spike_Arrange(raster_plot,trail_ids,condition_num = 1416):
    '''
    raster_plot is a single cell's raster.(data_dict['GoodUnitStrc']['Raster'])
    trail_ids is id of all response(data_dict['meta_data']['trial_valid_idx'])
    condition_num is the number of stimset.
    '''
    real_trail_ids = np.array([i for i in trail_ids if i != 0]) # remove 0 trails
    if len(raster_plot) != len(real_trail_ids):
        raise ValueError('Condition ID Mismatch, check please.')
    N_trail = len(real_trail_ids)//condition_num
    used_trail_ids = real_trail_ids[:N_trail*condition_num]
    id_lists = list(set(used_trail_ids))
    id_lists.sort()
    # arrange spike matrix. 
    arranged_spikes = np.zeros(shape=(len(used_trail_ids),raster_plot.shape[1]),dtype='f8')
    for i,cid in enumerate():
        
    

    return arranged_spikes,id_lists