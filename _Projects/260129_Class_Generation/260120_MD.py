'''
新的刺激集的class generation

'''



#%%
# from Cell_Class import Cell_Infos
import mat73
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


#%% ### 记得修改！！ ###

gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_260120_MaoDan_Doodle_AI_v260119_g0.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Doodle_v260119',
                            brain_areas=['MSB','ML'],
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

#%% Test of an NSD 1k dataset.

gn_folder = r'E:\#Preprocessed_Data\GoodUnits\nsd'
c_gn_path = 'GoodUnit_260120_MaoDan_NSD1000_g1.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='NSD_1k',
                            brain_areas=['MSB','ML'],
                            onset=250,
                            offset=150,
                            used_on=np.arange(160,320),
                            save_train=True,prepare_data=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}_{SRS.stimset}.joblib'), compress=7) 


#%%

gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_260120_MaoDan_Doodle_AI_v260119_g0.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Doodle_v260119',
                            brain_areas=['MSB','ML'],
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


#%%

gn_folder = r'E:\#Preprocessed_Data\GoodUnits\nsd'
c_gn_path = 'GoodUnit_260123_Maodan_NSD1000_g4_MSB_ML.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='NSD_1k',
                            brain_areas=['MSB','ML'],
                            onset=250,
                            offset=150,
                            used_on=np.arange(160,320),
                            save_train=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}_{SRS.stimset}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

#%%

gn_folder = r'E:\#Preprocessed_Data\GoodUnits\silct_new'
c_gn_path = 'GoodUnit_260123_Maodan_Doodle_AI_v260121_g3_ML_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Doodle_v260121',
                            brain_areas=['MSB','ML'],
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


#%%  FaCai ASB 


gn_folder = r'E:\#Preprocessed_Data\GoodUnits\silct_new'
c_gn_path = 'GoodUnit_26016_FaCai_Doodle_AI_v260121_g2_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Doodle_v260121',
                            brain_areas=['ASB'],
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


#%%


gn_folder = r'E:\#Preprocessed_Data\GoodUnits\silct_new'
c_gn_path = 'GoodUnit_260128_Facai_Doodle_AI_v260121_g5_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Doodle_v260121',
                            brain_areas=['ASB'],
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


#%%

gn_folder = r'E:\#Preprocessed_Data\GoodUnits\silct_new'
c_gn_path = 'GoodUnit_260130_MaoDan_Doodle_AI_v260121_g2_ML_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Doodle_v260121',
                            brain_areas=['MSB','ML'],
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


#%%

gn_folder = r'E:\#Preprocessed_Data\GoodUnits\silct_new'
c_gn_path = 'GoodUnit_260203_Maodan_Doodle_AI_v260121_g4_ML_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Doodle_v260121',
                            brain_areas=['MSB','ML'],
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

#%%

gn_folder = r'E:\#Preprocessed_Data\GoodUnits\metamer'
c_gn_path = 'GoodUnit_260203_Maodan_Mega_Metamer_v251104_g5_ML_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Mega_Metamer_v251104',
                            brain_areas=['MSB','ML'],
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

