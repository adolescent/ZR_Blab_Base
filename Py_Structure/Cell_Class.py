'''
##################################
251201 DECREPTED I find it unnecessary for this structure design.




This stuct will pack single cell info of 

This class is usually generated from response matrix, 
you need to provide :
    - FOB 
    - raw response matrix
These response should be unceilied and un averaged, raw response must be provided.

Return will be a class object, using method can get cell's ceiling index, preference, etc...

NOTE : Response here will not indicate any stim-related info, so for specific stim set, you need to find it by yourself. 

### This class is designed as a subclass of Site_Info, so direct usage might cause some untested problem.    
Input: Arranged series array for this cell

Output: Cell Info class.

'''
#%%
from Spike_Tools import *
import seaborn as sns
import os 
from Py_Structure.Info_Files.InfoLoader import Load_Info


class Cell_Infos(object):

    name='Recored cells of a single recording sites'

    def __init__(self,cell_matrix,stiminfo,masks,loc=(-1,-1),unit_type=-1,savepath='Default'):
        # cell matrix: It shall be a single cell  N_Repeat*N_Img*N_Timepoint
        # save path need 

        self.cell_matrix = cell_matrix
        self.N_Repeat = cell_matrix.shape[0]
        self.N_img = cell_matrix.shape[1]
        self.N_timepoint = cell_matrix.shape[2]
        self.avr_response = cell_matrix.mean(0)
        self.loc = loc
        self.unit_type=unit_type
        self.stimset = stiminfo
        self.masks = masks
        self.savepath = savepath

    def __len__(self): # length method will return number of graphs
        return self.N_img

    def __getitem__(self,img_key): # get item will retrun single img's response curve.
        return self.avr_response[img_key,:]

    def Noise_Ceiling(self,ceiling_method='all',time_range=(160,320),):
        # Method can be 'all' or 'fob', this will result 
        pass


    def Tuning_Calculate(self):# info is not build in, so not having it will not raise error.
        pass

    def cell_init(self):
        '''
        Do cell initialization for cell class,
        '''
        pass

#%% Testrun parts
import importlib
if __name__ == '__main__':

    a = Cell_Infos(np.ones(shape = (3,45,45)),'test',0)
    ot.Save_Variable(r'E:\#Sorting_tmp','test',a)

    # try path
    obj=ot.Load_Variable(r'E:\#Sorting_tmp\test.pkl')


#     #%%
#     wp=r'E:\#Preprocessed_Data\GoodUnits\GoodUnit_251105_ZhuangZhuang_Mega_Metamer_v251104_g1.mat'
#     data_dict = mat73.loadmat(wp,verbose=False)
#     raster_matrix = PSTH_From_Goodunit(wp,3680)
#     cell_matrix = raster_matrix[399,:,:,:] # use example cell for testing.
#     #%%
#     a = Cell_Infos(cell_matrix)

#     #%% test noise ceiling.
# from Py_Structure.Info_Files.InfoLoader import Load_tsv
    

