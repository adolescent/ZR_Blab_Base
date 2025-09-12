'''
Lite of generating fob index for tuning calculation.
'''

#%%
import pandas as pd
import numpy as np

#%% P1 FOB 72 index 
# 
fob_index = pd.DataFrame('',index=np.arange(72),columns=['Category'])
for i in range(72):
    if i <24:
        c_category = 'Body'
    elif i<48:
        c_category = 'Face'
    else:
        c_category = 'Object'
    fob_index.loc[i] = c_category

fob_index.to_csv('FOB72.csv')
## how to load?
# a=pd.read_csv('FOB72.csv',index_col=0)

#%% P2 STI 150 index
sti_info = pd.read_csv(r'Z:\Monkey_ephys\data\ZhangRui\_ML-vault\sti150_info.tsv',sep='\t',index_col=0)

fob_index = pd.DataFrame('',index=np.arange(150),columns=['Category'])

for i in range(150):
    c_slice = sti_info.iloc[i,:]
    cats = ['Face','Body','Scene','Object','Food']
    for j,c_cat in enumerate(cats):
        if c_cat in c_slice['FOB']:
            fob_index.loc[i]=c_cat
            break
fob_index.to_csv('STI150.csv')

