'''
small toolkit lite for convenient.

'''

import os
from mua_detection import MUA_Detector
import pandas as pd
from toolkits import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def Get_File_Name(path,file_type = '.tsv',keyword = ''):

    Name_Lists=[]
    for root, dirs, files in os.walk(path):
        for file in files:# walk all files in folder and subfolders.
            if root == path:# We look only files in root folder, subfolder ignored.
                if (os.path.splitext(file)[1] == file_type) and (keyword in file):# we need the file have required extend name and keyword contained.
                    Name_Lists.append(os.path.join(root, file))

    return Name_Lists

def D_Prime(disp_A,disp_B):
    '''
    Calculate d prime of 2 distributions. very widely used.
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

class Online_Viewer(object):
    name='Online tuning viewer'

    def __init__(self,
    wp,
    npx_path,
    thres=1.5,
    base_time = np.arange(75,125),
    onset_time = np.arange(150,250)
    ):

        self.wp = wp
        self.base_time = base_time
        self.onset_time = onset_time
        tsv_path = Get_File_Name(wp,'.tsv')[0]
        event_path = Get_File_Name(wp,'.csv','Onset_Times')[0]
        self.stim_info = pd.read_csv(tsv_path, sep='\t')
        event_time = pd.read_csv(event_path, sep=',')
        self.event_time = event_time[event_time['Trail_ID'] != 0].reset_index(drop=True)

        self.response_matrix = MUA_Detector(self.wp,npx_path,thres)
        print('Peak detection Done.')

    def PSTH_Calculator(self,before_time = 100,after_time=350):
        self.channel_num = self.response_matrix.shape[0]
        self.img_num = len(self.stim_info)
        time_points = after_time+before_time
        N_repeat = int(len(self.event_time)/self.img_num)
        self.psth = np.zeros(shape = (self.channel_num,self.img_num,time_points),dtype='f8')
        counter=np.zeros(self.img_num)
        # cycle event time for 
        for i in range(len(self.event_time)):
            c_event = self.event_time.iloc[i,:]
            c_times = np.arange((c_event['Onset_Time']-before_time),(c_event['Onset_Time']+after_time))
            c_id = c_event['Trail_ID']-1
            if counter[c_id] == N_repeat:
                continue
            else:
                self.psth[:,c_id,:] += self.response_matrix[:,c_times]
                counter[c_id] += 1

    def Global_dprime(self):
        self.d_primes = np.zeros(shape = (self.channel_num,self.img_num))
        for i in range(self.channel_num):
            for j in range(self.img_num):
                c_off = self.psth[i,j,self.base_time]
                c_on = self.psth[i,j,self.onset_time]
                self.d_primes[i,j]= D_Prime(c_off,c_on)
        self.d_primes = np.nan_to_num(self.d_primes)

    def Plot_Global(self,y_tick=30,x_tick=20):
        fig,ax = plt.subplots(ncols = 1,nrows =1,dpi=200,figsize=(3,5))
        plotable = self.d_primes
        p_std = plotable.std()
        p_mean = plotable.mean()
        plotable = np.clip(plotable,p_mean-3*p_std,p_mean+3*p_std)

        sns.heatmap(plotable,ax = ax,cmap='bwr')
        y_ticks = np.arange(0,plotable.shape[0],y_tick)
        x_ticks = np.arange(0,plotable.shape[1],x_tick)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
        for i,c_x in enumerate(x_ticks):
            ax.axvline(c_x,alpha=0.5,lw=1,color='gray',linestyle='--')
        for i,c_y in enumerate(y_ticks):
            ax.axhline(c_y,alpha=0.5,lw=1,color='gray',linestyle='--') 


    def Plot_Subtraction(self,channel_resp,posi_name=''):

        fig,ax = plt.subplots(ncols=1,nrows=2,dpi=200,figsize = (3,6),sharex=True, gridspec_kw={'height_ratios':[1,2]})
        ax[1].scatter(x=channel_resp,y=range(len(channel_resp)),s=10)
        ax[0].hist(channel_resp,bins=20)

        ax[0].axvline(x=self.d_primes.mean(),linestyle='--',c='gray')
        ax[1].axvline(x=self.d_primes.mean(),linestyle='--',c='gray')
        ax[1].set_xlabel(f'd prime of {posi_name}')
        fig.tight_layout()

        
    def Subtract(self,A_sets,B_sets,plot=True):
        A_sets_real = [self.clusts[i-1] for i in A_sets]
        B_sets_real = [self.clusts[i-1] for i in B_sets]
        print(f'A Set: {A_sets_real}')
        print(f'B Set: {B_sets_real}')
        face_ids = np.array(self.stim_info[self.stim_info['FOB'].isin(A_sets_real)].index)
        noface_ids = np.array(self.stim_info[self.stim_info['FOB'].isin(B_sets_real)].index)
        channel_face_resp = np.zeros(len(self.d_primes))
        for i in range(self.channel_num):
            channel_face_resp[i]=D_Prime(self.d_primes[i,noface_ids],self.d_primes[i,face_ids])
        channel_face_resp = np.nan_to_num(channel_face_resp)

        if plot == True:
            self.Plot_Subtraction(channel_face_resp,A_sets_real)

        return channel_face_resp
    
    def Plot_Sorted_Global(self,channel_resp,y_tick=30,x_tick=20):

        sorted_indices = np.argsort(channel_resp)[::-1]
        sorted_data = self.d_primes[sorted_indices]

        fig,ax = plt.subplots(ncols = 1,nrows =1,dpi=200,figsize=(3,5))
        plotable = sorted_data
        p_std = plotable.std()
        p_mean = plotable.mean()
        plotable = np.clip(plotable,p_mean-3*p_std,p_mean+3*p_std)
        sns.heatmap(plotable,ax = ax,cmap='bwr')
        y_ticks = np.arange(0,plotable.shape[0],y_tick)
        x_ticks = np.arange(0,plotable.shape[1],x_tick)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
        for i,c_x in enumerate(x_ticks):
            ax.axvline(c_x,alpha=0.5,lw=1,color='gray',linestyle='--')
        for i,c_y in enumerate(y_ticks):
            ax.axhline(c_y,alpha=0.5,lw=1,color='gray',linestyle='--') 


    def Process(self):
        self.clusts = list(set(self.stim_info['FOB']))
        self.clusts.sort()
        print(f'All Clusts:{self.clusts}')
        self.PSTH_Calculator()
        self.Global_dprime()
        self.Plot_Global()
        
#%%
if __name__ == '__main__':
    wp=r'D:\#Data\Loc_Example\test_wp'
    npx_path = r'D:\#Data\Loc_Example\npx_root_decoy\NPX_MD241029_exp_g0'

    ov = Online_Viewer(wp,npx_path)
    ov.Process()
    ov.Plot_All()

    #%% d prime graphs 
    print(ov.clusts)
    ov.Subtract([3],[1,2,4,5,6],plot=True)