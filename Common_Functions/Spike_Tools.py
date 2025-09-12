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
import mat73
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


def D_Prime(disp_A,disp_B):
    '''
    Calculate d prime of 2 distributions. very widely used.
    A is base, will calculate B-A.
    '''
    mu_n = np.mean(disp_A)
    mu_s = np.mean(disp_B)
    var_n = np.var(disp_A, ddof=1)  # ddof=1 用于样本方差
    var_s = np.var(disp_B, ddof=1)

    # 计算合并标准差
    n_n = len(disp_A)
    n_s = len(disp_B)
    pooled_var = ((n_n - 1) * var_n + (n_s - 1) * var_s) / (n_n + n_s - 2)
    pooled_std = np.sqrt(pooled_var)

    # 计算 d prime
    d_prime = (mu_s - mu_n) / pooled_std

    return d_prime


def PSTH_From_Goodunit(good_unit_path,img_num = 1416):

    data_dict = mat73.loadmat(good_unit_path,verbose=False) #mute warning.
    trail_info = data_dict['meta_data']['trial_valid_idx']  
    raster_info = data_dict['GoodUnitStrc']['Raster']

    # get basic psth shape.
    cellnum = len(raster_info)
    time_points = (raster_info[0]).shape[1]
    cond_num = raster_info[0].shape[0]//img_num
    PSTH = np.zeros(shape = (cellnum,cond_num,img_num,time_points),dtype='u1')

    # cycle cell for response.
    for i in tqdm(range(cellnum)):
        cc_response = Spike_Arrange(raster_info[i],trail_info,img_num)
        PSTH[i,:,:,:] = cc_response

    return PSTH


def odd_end_ceiling(fob_dataset,used_time = np.arange(150,250)):
    '''
    fob_dataset contain used fob part of series, must be in shape
    (N_cell*N_repeat*N_FOB*N_Timepoint)

    Then we will calculate average response of each cell in odd&end runs, return a repeat consistency for data points.
    '''
    n_cell = fob_dataset.shape[0]
    ceiling_index = np.zeros(n_cell)

    fob_onset = fob_dataset[:,:,:,used_time].sum(-1) # get total firing rate in onset time
    fob_odd = fob_onset[:, ::2, :].mean(1)
    fob_end = fob_onset[:, 1::2, :].mean(1)

    for i in range(n_cell):
        c_r,_ = stats.pearsonr(fob_odd[i,:],fob_end[i,:])
        ceiling_index[i] = c_r
    ceiling_index = np.nan_to_num(ceiling_index)


    return ceiling_index
    

def Redplot(raw_data,base=np.arange(75,125),onset = np.arange(150,250)):
    '''
    Calculate response d prime for data points in psth matrix.
    raw_data must be in shape(N_cellxN_framexN_time),remember to average different trail.
    
    '''
    n_cell = raw_data.shape[0]
    n_frame = raw_data.shape[1]
    redplot = np.zeros(shape=(n_cell,n_frame))
    for i in range(n_cell):
        for j in range(n_frame):
            c_base = raw_data[i,j,base]
            c_resp = raw_data[i,j,onset]
            c_d = D_Prime(c_base,c_resp)
            redplot[i,j]=c_d
    redplot = np.nan_to_num(redplot)

    return redplot

def Calculate_Cell_Tunings(response,infos,base=np.arange(75,125),onset = np.arange(150,250)):
    '''
    response must be in shape [N_cellxN_FOBxN_Timepoints].
    info must be a pandas dataframe, providing category of each fob id.
    NOTE info and response's FOB MUST be the same length, so if you repeat more than 1 time, provide longer info.
    '''
    used_cats = list(set(infos['Category']))
    n_cell = response.shape[0]
    tuning_frame = pd.DataFrame(columns=['Cell','Contra','D_Prime'],index=np.arange(len(used_cats)*2*n_cell))
    counter=0

    for i,c_cat in enumerate(used_cats):
        c_ids = np.array(infos[infos['Category']==c_cat].index)
        mask = np.ones(response.shape[1], dtype=bool)
        mask[c_ids] = False
        target_resp = response[:,c_ids,:]
        contra_resp = response[:,mask,:]
        # calculate contra vs pre first.
        # target_on = target_resp[:,:,onset].reshape((n_cell,-1))
        # target_off = target_resp[:,:,base].reshape((n_cell,-1))
        target_on = target_resp[:,:,onset].mean(-1)
        target_off = target_resp[:,:,base].mean(-1)
        contra_on = contra_resp[:,:,onset].mean(-1)

        for j in range(n_cell):
            c_d = D_Prime(target_off[j,:],target_on[j,:])
            tuning_frame.loc[counter,:] = [j,c_cat+'_Onset',np.nan_to_num(c_d)]
            counter+=1
            c_d = D_Prime(contra_on[j,:],target_on[j,:])
            tuning_frame.loc[counter,:] = [j,c_cat+'_Contra',np.nan_to_num(c_d)]
            counter+=1


    return tuning_frame