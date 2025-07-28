'''
This script will transform Good unit data into python-readable format.
At the same time will align unit data to different img onset, getting response curve toward each cell.

'''


#%%

from Common_Functions.OS_Tools import *
# import scipy
import numpy as np
import h5py
import matplotlib.pyplot as plt

gn_path = r'D:\#Data\silct\GoodUnit_250614_ZhuangZhuang_silct_npx_1416_g4.mat'
# gn_raw = scipy.io.loadmat(gn_path)
f = h5py.File(gn_path,'r')
arrays = {}
for k, v in f.items():
    arrays[k] = np.array(v)
print(arrays.keys())
#%%
data_ref = f.get('GoodUnitStrc/response_matrix_img')
data = np.array(f[data_ref[3][0]]).T # this will return only reference of this data. We need to use this pointer to get data.
# plt.imshow(data[1200:,:])
plt.imshow(data)
# H5_File_Tree(data)


#%% another version, this works, but a little buggy.
import mat73
data_dict = mat73.loadmat(gn_path)
#%%
# raw_data = data_dict['global_params']['PsthRange']
raw_data = data_dict['GoodUnitStrc']['response_matrix_img']
# raw_data[0].shape
plt.imshow(raw_data[3])
# print(raw_data.keys())
# %%
plt.imshow(raw_data[20])
