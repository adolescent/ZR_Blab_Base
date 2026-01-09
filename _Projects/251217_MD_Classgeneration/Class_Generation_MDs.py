

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
save_path = r'E:\#Preprocessed_Data\SiteClass'



#%% 

gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251212_MaoDan_Metamer_Singlebubble_v251107_4540_g2_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer_Singlebubble_v251107',
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



#%% 

gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251215_Maodan_Metamer_Singlebubble_v251107_4540_g2.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer_Singlebubble_v251107',
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

#%%

gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251226_MD_Mega_Metamer_v251104_g2_MSB_MF.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Mega_Metamer_v251104',
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

#%%


