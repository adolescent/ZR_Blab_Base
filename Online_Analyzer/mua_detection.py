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
import csv
# from pathlib import Path


def MUA_Detector(wp,npx_path,thres = 1.5):

    # keyword = 'NPX'
    # npx_path = []
    # for root, dirs, files in os.walk(wp):
    #     if root == wp:
    #         for dir_name in dirs:
    #             if keyword in dir_name:
    #                 npx_path.append(os.path.join(root, dir_name))
    # # npx_path = os.path.join(wp,'NPX_MD241029_exp_g0')
    # npx_path = npx_path[0]
    # tsv_path = os.path.join(wp,'WordLocalizer_info.tsv')
    # bhv_path =os.path.join(wp,'241026_MaoDan_YJ_WordLOC.bhv2')
    # # if raw exist, just load it in.
    bin_path = []
    for root, dirs, files in os.walk(wp):
        if root == wp:
            for dir_name in dirs:
                if 'Bin_File' in dir_name:
                    bin_path.append(os.path.join(root, dir_name))
    if bin_path != []:
        print('Bin format already saved.')
        rec3 = si.load(bin_path[0])
    else:
        print('Preprocessing Data, might be a little slow.')
        # get raw data and 
        stream_names, stream_ids = si.get_neo_streams('spikeglx',npx_path)
        print(stream_names)
        raw_rec = si.read_spikeglx(npx_path, stream_name='imec0.ap', load_sync_channel=False)
        raw_rec.get_probe().to_dataframe()
        # eazy process

        # remove bad channel id
        rec3=raw_rec
        # bad_channel_ids, channel_labels = si.detect_bad_channels(raw_rec)
        # rec3 = raw_rec.remove_channels(bad_channel_ids)
        # print('Preprocess Real Data, bad_channel_ids', bad_channel_ids)

        ## high pass filter
        print('Preprocess Real Data, High Pass')
        rec3 = si.highpass_filter(recording=rec3, freq_min=300.)
        # rec3 = si.bandpass_filter(recording=rec3,freq_min=300)


        ## pahse shift
        # print('Preprocess Real Data, phase shift')
        # rec3 = si.phase_shift(rec3)

        ## common reference
        print('Preprocess Real Data, CAR')
        rec3 = si.common_reference(rec3, operator="median", reference="global")
        # rec = rec3
        #% save pre-processed data.
        # # rasters = traces0>(traces0.std()*mua_thres)
        job_kwargs = dict(chunk_duration='1s', progress_bar=True,overwrite=True)
        rec3 = rec3.save(folder=os.path.join(wp,'Bin_File'), format='binary',**job_kwargs)

    # try build-in threshold detector.

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks


    # job_kwargs = dict(n_jobs=8, chunk_duration='1s', progress_bar=True)
    # noise_levels = si.get_noise_levels(rec3,**job_kwargs)
    noise_levels = si.get_noise_levels(rec3)

    job_kwargs = dict(progress_bar=True,chunk_duration='10s')
    peaks = detect_peaks(rec3, noise_levels=noise_levels,method = 'locally_exclusive',
                        detect_threshold=thres,**job_kwargs)

    np.save(os.path.join(wp,'All_Peaks.npy'),peaks)

    ##  rearrange peak, so can we get spike time of each channel.
    n_channel = rec3.get_num_channels()
    n_frame = rec3.get_num_samples()
    fps = rec3.get_sampling_frequency()
    capture_ms = int(n_frame/(fps/1000))+1
    response_matrix = np.zeros(shape = (n_channel,capture_ms))
    response_matrix = response_matrix.astype('int16')

    ## cycle peak for response matrix.
    print('Peak detection done, transform data into response matrix...')
    for i in range(len(peaks)):
        cc_peak = peaks[i]
        ch = cc_peak[1]
        time = cc_peak[0]
        time_ms = int(time/(fps/1000))
        response_matrix[ch,time_ms]+=1
    # save as csv for matlab usage.
    # filename = os.path.join(wp,"Peak_Info.csv")
    # sns.heatmap(response_matrix[:,0:5000])
    # fig,ax = plt.subplots(ncols=1,nrows=1,dpi=)


    return response_matrix
#%%
if __name__ == '__main__':
    wp = r'D:\#Data\Loc_Example\Example_Data_FOB_RAW' # dir of NPX data
    npx_path = r'D:\#Data\Loc_Example\Example_Data_FOB_RAW\NPX_MD241029_exp_g0'
    MUA_Detector(wp,thres = 1.5)


