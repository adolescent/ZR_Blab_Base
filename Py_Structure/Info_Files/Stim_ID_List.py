'''
This script save info with no fob, making it easier to load stim info qucik.
'''
#%%

import numpy as np
import pandas as pd


class Stim_ID(object):

    def __init__(self,stim_type='Metamer_Raw'):

        if stim_type == 'Metamer_Raw':
            self.Metamer_Info_Raw()
        
        # return self.Stim_Conditions

    def Metamer_Info_Raw(self):
        img_indices = np.tile(np.arange(1, 41), 25)
        # Generate Shuffle Level column: for each of 0-4, repeat 40 times; the whole 0-4 block is repeated 25 times
        shuffle_levels = np.tile(np.repeat(np.arange(5), 40), 5)
        # Ensure arrays are the correct length
        assert len(img_indices) == 1000
        assert len(shuffle_levels) == 1000
        self.Stim_Conditions = pd.DataFrame({
            'Img_Index': img_indices,
            'Shuffle_Level': shuffle_levels
        })
        # return self.Stim_Conditions