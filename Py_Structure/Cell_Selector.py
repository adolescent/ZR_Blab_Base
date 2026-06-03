'''
Give in Cell Class (SRS file), ceiling thres, tuning preference, DPrime requirements, this will return selected dicts, including:


1.FOB Response of current selected cells in loc(averaged FOB)
1.AVR Response of loc
2.AVR PSTH (time included) of loc
3.Preso Trail-by-Trial Response of loc
4.Preso Trail-by-Trial PSTH (time included) of loc
5.Raw PSTH of current case
NOTE All for selected cells only.

'''
#%% imports

from Py_Structure.Info_Files.InfoLoader import Select_Cell_Info
from Py_Structure.Struct_Funcs import Single_Recording_Site
import joblib as JL
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
def Stim_Cell_Rearrange(Cell_Class,
                        ceiling_thres=0.3,
                        prefer='Body',
                        dp_thres=0.5,
                        stim_type='Doodle',
                        time_start= 160,
                        time_end = 320
                        ):

    graph_infos = Select_Cell_Info(stim_info=stim_type)
    c_stimset = Cell_Class.stimset
    c_stimset_info = graph_infos[c_stimset]


    selected_cell_ids,tuned_psth = Cell_Class.Cell_Selection(ceiling=ceiling_thres,prefer=prefer,dp_thres=dp_thres)
    ## generate fob first.
    fob_lengths = {'STI150':150,'Wordloc':180,'FOB72':72}
    fob_style = c_stimset_info['FOB']['style']
    fob_ids = c_stimset_info['FOB']['id']
    n_repeat = len(fob_ids)//fob_lengths[fob_style]
    ## for average response, we can use tuned_psth
    avr_redplot = tuned_psth[:,:,time_start:time_end].sum(-1)
    # 根据n_repeat对fob_rsp进行reshape，并对不同repeat取平均
    arr = avr_redplot[:, fob_ids]
    if n_repeat > 1:
        arr = arr.reshape(arr.shape[0], n_repeat, -1)
        fob_rsp = arr.mean(axis=1)
    else:
        fob_rsp = arr

    ## then generate average response.
    data_ids = c_stimset_info['Data']
    raw_psth = Cell_Class.raw_psth[selected_cell_ids,:,:]
    avr_rsp = avr_redplot[:,data_ids]
    avr_psth = raw_psth[:,:,data_ids,:].mean(1)

    ## average response and psth by trail, for further decoding&encoding.
    avr_rsp_by_trail = raw_psth[:,:,data_ids,time_start:time_end].mean(-1)
    avr_psth_by_trail = raw_psth[:,:,data_ids,:]

    return fob_rsp,avr_rsp,avr_psth,avr_rsp_by_trail,avr_psth_by_trail,raw_psth



def Pseudo_Generator(avr_rsp_by_trail, N_fold=5, N_pseudo=10, seed=0):
    """LOO CV: each fold holds out one repeat as test (N_Cell, N_Image);
    train = N_pseudo pseudo matrices from the other repeats (no trial overlap with test)."""
    rsp = np.asarray(avr_rsp_by_trail)
    n_cell, n_repeat, n_img = rsp.shape
    n_fold = min(N_fold, n_repeat)
    rng = np.random.default_rng(seed)

    testsets, trainsets = [], []
    for fold in range(n_fold):
        test_idx = fold
        train_rsp = rsp[:, np.arange(n_repeat) != test_idx, :]
        testsets.append(rsp[:, test_idx, :])
        pseudos = []
        for _ in range(N_pseudo):
            pick = rng.integers(0, train_rsp.shape[1], size=n_img)
            pseudos.append(train_rsp[:, pick, np.arange(n_img)])
        trainsets.append(pseudos)
    return testsets, trainsets



def Pseudo_Generator_PSTH(avr_psth_by_trail, N_fold=5, N_pseudo=10, seed=0):
    """Same as Pseudo_Generator; shapes include time: test (N_Cell, N_Image, N_Time)."""
    rsp = np.asarray(avr_psth_by_trail)
    n_cell, n_repeat, n_img, n_time = rsp.shape
    n_fold = min(N_fold, n_repeat)
    rng = np.random.default_rng(seed)

    testsets, trainsets = [], []
    for fold in range(n_fold):
        test_idx = fold
        train_rsp = rsp[:, np.arange(n_repeat) != test_idx, :, :]
        testsets.append(rsp[:, test_idx, :, :])
        pseudos = []
        for _ in range(N_pseudo):
            pick = rng.integers(0, train_rsp.shape[1], size=n_img)
            pseudos.append(train_rsp[:, pick, np.arange(n_img), :])
        trainsets.append(pseudos)
    return testsets, trainsets

#%% test run
if __name__ == '__main__':
    SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\Doodle\MSB\260130_MaoDan_PV_OE_doodle_MSB_ML_Doodle_v260121.joblib')
    a,b, = SRS.Cell_Selection(ceiling=0.3,prefer='face',dp_thres=0.5)


