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
from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site


save_parh = r'E:\#Preprocessed_Data\SiteClass'
#%%
c_gn_path = r'E:\#Preprocessed_Data\GoodUnits\GoodUnit_251110_ZhuangZhuang_Metamer_Singlebubble_v251107_4540_g1_AL_ASB.mat'
SRS = Single_Recording_Site(gn_path=c_gn_path,
                            stimset='Metamer_Singlebubble_v251107',
                            brain_areas=['AL','ASB'],
                            onset=300,
                            offset=200,
                            used_on=np.arange(160,320),
                            save_train=True
                            )
# SRS.__dict__.keys()
#%%

