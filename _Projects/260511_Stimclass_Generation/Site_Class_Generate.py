'''
Generate my prefered data structure from good unit.

This also quicker on data loader.

'''

#%%
# from Cell_Class import Cell_Infos
import mat73
# from numpy import False_
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site

from Common_Functions.Useful_Plotter import *
import warnings
warnings.filterwarnings("ignore")

save_path = r'E:\#Preprocessed_Data\SiteClass'

#%%
c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\anagram_jigsaw\GoodUnit_260424_JianJian_Anagram_Jigsaw_v260227_g4_AL_ASB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Anagram_Jigsaw_v260227',
                            brain_areas=['ASB','AL'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))

#%%
c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\silct_new\GoodUnit_260424_JianJian_Doodle_AI_v260121_g5_AL_ASB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Doodle_v260121',
                            brain_areas=['ASB','AL'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))


#%%

c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\metamer\GoodUnit_260424_JianJian_Metamer_NSD_FOB_v260420_g3_AL_ASB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer_NSD',
                            brain_areas=['ASB','AL'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))

#%%
c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\anagram_jigsaw\GoodUnit_260504_DiQue_Anagram_Jigsaw_v260227_g5_AL_ASB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Anagram_Jigsaw_v260227',
                            brain_areas=['ASB','AL'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))

#%% 260506 DQ ALO

c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\anagram_jigsaw\GoodUnit_260506_DiQue_Anagram_Jigsaw_v260227_g5_ALO.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Anagram_Jigsaw_v260227',
                            brain_areas=['ALO'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation
pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))



#%%
c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\silct_new\GoodUnit_260506_DiQue_Doodle_AI_v260430_g4_ALO.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Doodle_AI_v260430',
                            brain_areas=['ALO'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation
#
pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['spiky','stubby','word']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))


#%% ################### 260509 MD MSB-ML，strange stimsets #################

c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\ani_texform\GoodUnit_260509_MaoDan_AnimSizeTexform_v260507_g7_ML_MSB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Ani_Texform_v260507',
                            brain_areas=['ML','MSB'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=False
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation
#
pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]

fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))


#%%


c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\DQInva\GoodUnit_260509_MaoDan_DQInva_v260508_g8.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='DQInva_v260508',
                            brain_areas=['ML','MSB'],
                            onset=800,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=False
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation
#
pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]

fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))


#%%



c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\DQInva_Video\GoodUnit_260509_Maodan_short_videos_g10_ML_MSB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='DQInva_Video_v260508',
                            brain_areas=['ML','MSB'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(1060,1220), # video from -1000
                            save_train=False,
                            video=True,img_size= 306,
                            txt_path=r'Z:\Monkey\Stimuli\ZR\Video_DQInva_260508\Video_DQInva_260508.txt'
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation
#
pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]

fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))


#%%


c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\anagram_jigsaw\GoodUnit_260528_JianJian_Anagram_Jigsaw_v260227_g0_MSB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Anagram_Jigsaw_v260227',
                            brain_areas=['ML','MSB'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320), 
                            save_train=False,
                            video=False
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}_{SRS.stimset}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation
#
pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]

fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))

#%%


c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\ani_texform\GoodUnit_260604_JJ_AnimSizeTexform_v260507_g4_ML_MSB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Ani_Texform_v260507',
                            brain_areas=['ML','MSB'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=False
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation
#
pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]

fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))


#%%


c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\DQInva\GoodUnit_260604_JJ_DQInva_v260508_g3_ML_MSB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='DQInva_v260508',
                            brain_areas=['ML','MSB'],
                            onset=800,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=False
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')
# Plot FOB for class estimate generation
#
pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]

fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))




#%%
if __name__ == '__main__':
    test_SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\anagram_jigsaw\260528_JianJian_PV_OE_JigSaw_MSB_ML_Anagram_Jigsaw_v260227.joblib')

    