'''
Generate my prefered data structure from good unit.

This also quicker on data loader.

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
save_path = r'E:\#Preprocessed_Data\SiteClass'

#%%
c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\GoodUnit_251108_ZhuangZhuang_Metamer_Singlebubble_v251107_4540_g2_MSB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer_Singlebubble_v251107',
                            brain_areas=['MSB'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )
# SRS.__dict__.keys()

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')


#%% Plot FOB for class estimate generation

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))

#%%
'''

Another points.

'''
c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\GoodUnit_251105_ZhuangZhuang_Mega_Metamer_v251104_g1.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Mega_Metamer_v251104',
                            brain_areas=['AL','ASB'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))


#%%
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251103_JianJian_Mega_Metamer_v250920_g1_AL_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Mega_Metamer_v250920',
                            brain_areas=['AL','ASB'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))


#%%
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251031_JianJian_Metamer_Cut_v251011_g3_AL_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer_Cut_v251011',
                            brain_areas=['AL','ASB'],
                            onset=350,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))

#%%
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251027_JianJian_Metamer_P4_C4321_Object_STI150_1300_g5_ML.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer1300',
                            brain_areas=['ML'],
                            onset=200,
                            offset=150,
                            used_on=np.arange(160,320),
                            save_train=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))

#%%
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251020_Jianjian_Metamer_P4_C4321_Object_STI150_1300_g5_AL.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer1300',
                            brain_areas=['AL'],
                            onset=200,
                            offset=150,
                            used_on=np.arange(160,320),
                            save_train=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))


#%%
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251014_Zhuangzhuang_Metamer_Cut_v251011_g8_AL_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer_Cut_v251011',
                            brain_areas=['AL','ASB'],
                            onset=200,
                            offset=150,
                            used_on=np.arange(160,320),
                            save_train=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))

#%%
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251013_Jianjian_Metamer_P4_C4321_Object_STI150_1300_g6_AL.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer1300',
                            brain_areas=['AL'],
                            onset=200,
                            offset=150,
                            used_on=np.arange(160,320),
                            save_train=True
                            )

JL.dump(SRS,ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.joblib'), compress=7) # save compressed file, 8 compress for speed and storage.
# SRS = JL.load(r'E:\#Preprocessed_Data\SiteClass\251110_ZhuangZhuang_PV_OE_metamer_single_bubble_AL_ASB.joblib')

pivot_dp = pd.pivot(SRS.Cell_FOB_DPrimes,columns='Category',index='Cell',values='D_Prime')
pivot_dp = pivot_dp[['Body','Face','Object']]
fig = Triangle_FOB(pivot_dp,1,800,600,label=False)
# fig.write_image("ternary_plot.png")
fig.write_html(ot.Join(save_path,f'{SRS.site_name}_{'_'.join(SRS.brain_areas)}.html'))

#%%
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250927_ZhuangZhuang_Mega_Metamer_v250920_g1_AL_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Mega_Metamer_v250920',
                            brain_areas=['AL','ASB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250927_ZhuangZhuang_silct_npx_1416_g4_AL_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['AL','ASB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250925_ZhuangZhuang_silct_npx_1416_g3_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['MSB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250925_ZhuangZhuang_Mega_Metamer_v250920_g2_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Mega_Metamer_v250920',
                            brain_areas=['MSB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250923_Zhuangzhuang_Mega_Metamer_v250920_g3_AL_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Mega_Metamer_v250920',
                            brain_areas=['AL','ASB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250922_JianJian_Mega_Metamer_v250920_g0_AL_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Mega_Metamer_v250920',
                            brain_areas=['AL','ASB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250910_JianJian_Metamer_P4_C4321_Object_STI150_1300_g1_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer1300',
                            brain_areas=['AL','ASB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250902_ZhuangZhuang_Metamer_P4_C4321_Object_STI150_1300_g2_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer1300',
                            brain_areas=['MSB'],
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250828_JianJian_Metamer_Pool4_C4321_Object_1k+FOB_g6_AL_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer1072',
                            brain_areas=['AL','ASB'],
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250722_JianJian_silct_npx_1416_g1_AL_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['AL','ASB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250616_ZhuangZhuang_silct_npx_1416_g5_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['MSB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250614_ZhuangZhuang_silct_npx_1416_g4_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['MSB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250423_JianJian_silct_npx_1416_g7_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['AL','ASB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250411_JianJian_silct_npx_1416_g0_MSB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['MSB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250824_ZhuangZhuang_silct_npx_1416_g2.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['AL','ASB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_250325_ZhuangZhuang_silct_npx_1416_g4_ASB.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Silct1416',
                            brain_areas=['AL','ASB'],
                            onset=200,
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
'''

Another points.

'''
gn_folder = r'E:\#Preprocessed_Data\GoodUnits'
c_gn_path = 'GoodUnit_251202_ZhuangZhuang_Metamer_Singlebubble_v251107_4540_g1.mat'
c_gn_path = ot.Join(gn_folder,c_gn_path)
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer_Singlebubble_v251107',
                            brain_areas=['MSB_like'],
                            onset=300,
                            offset=200,
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












