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
# import copy


def Spike_Arrange(raster_plot,trail_ids,condition_num = 1416,keep_all = False):
    '''
    raster_plot is a single cell's raster.(data_dict['GoodUnitStrc']['Raster'])
    trail_ids is id of all response(data_dict['meta_data']['trial_valid_idx'])
    condition_num is the number of stimset.
    keep_all = False will delete unfinished last trail.
    '''
    real_trail_ids = np.array([i for i in trail_ids if i != 0]).astype('i8') # remove 0 trails
    if len(raster_plot) != len(real_trail_ids):
        raise ValueError('Condition ID Mismatch, check please.')
    # get max repeat num of given id.
    max_trail = np.bincount(real_trail_ids)[1:].max() # max repeat time of each index.
    arranged_spikes = np.full((max_trail, condition_num,len(raster_plot[0])), np.nan)

    counters = np.zeros(condition_num,dtype='i4') # keep in mind of start from 0 or 1 = =
    for i,c_id in enumerate(real_trail_ids):
        arranged_spikes[counters[c_id-1],c_id-1,:]=raster_plot[i]
        counters[c_id-1] += 1 # add to 1 loc.
    if keep_all == False:
        N_full = len(real_trail_ids)//condition_num
        arranged_spikes = arranged_spikes[:N_full,:,:]
    return arranged_spikes


