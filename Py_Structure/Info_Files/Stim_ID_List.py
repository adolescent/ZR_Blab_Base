'''
This script save info with no fob, making it easier to load stim info qucik.
'''
#%%

import numpy as np
import pandas as pd


class Stim_ID(object):

    def __init__(self,stim_type='Metamer_Raw'):

        print(f'Loading {stim_type} info...')

        if stim_type == 'Metamer_Raw':
            self.Metamer_Info_Raw()
        elif stim_type == 'Anagram_Jigsaw':
            self.Anagram_Jigsaw_Info_Raw()
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

    def Anagram_Jigsaw_Info_Raw(self):
        total_repeats = 3
        styles = np.arange(1, 6)  # 5 styles
        pairs_per_style = 20
        ab_labels = ['A', 'B']
        pair_type_labels = ['FF', 'FA', 'AI', 'II']

        records = []

        # Each repeat has 200 images = 100 pairs = 5 styles * 20 pairs/style
        for _ in range(total_repeats):
            for pair_id in range(1, pairs_per_style + 1):
                # Pair_Tyle is tied to object pair (Img_Index), not style.
                # Each Img_Index occupies 10 rows (5 styles * A/B), all with same Pair_Tyle.
                pair_type = pair_type_labels[(pair_id - 1) % 4]
                for style in styles:
                    for ab in ab_labels:
                        records.append({
                            'Img_Index': pair_id,
                            'Style': style,
                            'Pair_Tyle': pair_type,
                            'AB': ab
                        })

        self.Stim_Conditions = pd.DataFrame(records)

        assert len(self.Stim_Conditions) == 600

#%%
if __name__ == '__main__':
    Stim_ID = Stim_ID('Anagram_Jigsaw')
    stim_info = Stim_ID.Stim_Conditions
    stim_info.iloc[:40]