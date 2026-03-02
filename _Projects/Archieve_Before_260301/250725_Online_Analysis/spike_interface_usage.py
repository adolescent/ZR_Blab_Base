'''
This script will try to use spike interface reading raw data, and here we only detect peak, skip sorting for speed.
The aim here is generate d prime graph online, as quick as possible.

'''

#%% import and basic path
# import matlab.engine
import spikeinterface.full as si
# from spikeinterface.preprocessing import correct_motion
import numpy as np
# import matplotlib.pyplot as plt
import os
# from pathlib import Path

# wp = r'D:\#Data\Loc_Example\Example_Data_FOB' # dir of NPX data
def MUA_Detector(wp):

    keyword = 'NPX'
    npx_path = []
    for root, dirs, files in os.walk(wp):
        if root == wp:
            for dir_name in dirs:
                if keyword in dir_name:
                    npx_path.append(os.path.join(root, dir_name))
    # npx_path = os.path.join(wp,'NPX_MD241029_exp_g0')
    npx_path = npx_path[0]
    # tsv_path = os.path.join(wp,'WordLocalizer_info.tsv')
    # bhv_path =os.path.join(wp,'241026_MaoDan_YJ_WordLOC.bhv2')


    #%% get raw data and 
    stream_names, stream_ids = si.get_neo_streams('spikeglx',npx_path)
    print(stream_names)
    raw_rec = si.read_spikeglx(npx_path, stream_name='imec0.ap', load_sync_channel=False)
    raw_rec.get_probe().to_dataframe()
    #%% eazy process

    ## remove bad channel id
    bad_channel_ids, channel_labels = si.detect_bad_channels(raw_rec)
    rec3 = raw_rec.remove_channels(bad_channel_ids)
    print('Preprocess Real Data, bad_channel_ids', bad_channel_ids)

    ## high pass filter
    print('Preprocess Real Data, High Pass')
    # rec3 = si.highpass_filter(recording=rec3, freq_min=300.)
    rec3 = si.bandpass_filter(recording=rec3,freq_min=300)

    ## pahse shift
    print('Preprocess Real Data, phase shift')
    rec3 = si.phase_shift(rec3)

    ## common reference
    print('Preprocess Real Data, CAR')
    rec3 = si.common_reference(rec3, operator="median", reference="global")
    # rec = rec3
    #%% save pre-processed data.
    # # rasters = traces0>(traces0.std()*mua_thres)
    job_kwargs = dict(chunk_duration='0.5s', progress_bar=True,overwrite=True)
    rec3 = rec3.save(folder='preprocess', format='binary',**job_kwargs)

    #%% try build-in threshold detector.

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks


    # job_kwargs = dict(n_jobs=8, chunk_duration='1s', progress_bar=True)
    # noise_levels = si.get_noise_levels(rec3,**job_kwargs)
    noise_levels = si.get_noise_levels(rec3)

    job_kwargs = dict(progress_bar=True,chunk_duration='10s')
    peaks = detect_peaks(rec3, noise_levels=noise_levels,method = 'by_channel',
                        detect_threshold=4,**job_kwargs)
    np.save(os.path.join(wp,'All_Peaks.npy'),peaks)

    return True