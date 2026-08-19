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
from Common_Functions.Useful_Plotter import *
import joblib as JL
import warnings
warnings.filterwarnings('ignore')

save_path = r'E:\#Preprocessed_Data\SiteClass'


#%%
gn_folder = r'E:\#Preprocessed_Data\GoodUnits\anagram_jigsaw'
c_gn_path = 'GoodUnit_260708_Faladi_Anagram_Jigsaw_v260227_g4_V4.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Anagram_Jigsaw_v260227',
                            brain_areas=['V4'],
                            onset=250,
                            offset=150,
                            used_on=np.arange(160,320),
                            save_train=True,video=False
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}_{SRS.stimset}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}_{SRS.stimset}.html'))
#%% ########  VIDEO VERSIONS HERE.
gn_folder = r'E:\#Preprocessed_Data\GoodUnits\DQInva_Video'
c_gn_path = 'GoodUnit_260604_JJ_short_videos_g5_ML_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='DQInva_Video_v260508',
                            brain_areas=['MSB','ML'],
                            onset=250,
                            offset=150,
                            used_on=np.arange(1050,1220),# as 1000ms is onset.
                            save_train=True,video=True,txt_path=r'Z:\Monkey_ephys\data_nas3\Zhangrui\_Processed\All_Processed_File\260604_JJ_MSB\dqinva_video\Video_DQInva_260508.txt'
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}_{SRS.stimset}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}_{SRS.stimset}.html'))

#%% test fob


a = SRS.raw_redplot
sns.heatmap(a/a.max(1,keepdims=1),cmap='bwr',vmax=1,center=0)


