#%%
'''
After generation of Goodunit.mat, you need to use this function to generate the site class.

Further analysis are mostly based on site class.

'''

#%%
import OS_Tools as ot
from Py_Structure.Struct_Funcs import Single_Recording_Site
from joblib import dump, load
import pandas as pd
import numpy as np
from Spike_Tools import Triangle_FOB
import joblib as JL

save_path = r'E:\#Preprocessed_Data\SiteClass'


#%%
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251215_Maodan_silct_npx_1416_g4.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['MSB','MF'],
                            onset=250,
                            offset=150,
                            used_on=np.arange(160,320),
                            save_train=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}_{SRS.stimset}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}_{SRS.stimset}.html'))