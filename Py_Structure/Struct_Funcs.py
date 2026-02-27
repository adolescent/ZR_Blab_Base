'''

This script contain severay public functions for structure generation, only public method contained here, to improve readability of private methods.


Final expected results:

Input Goodunit path, fob range and fob tsv:

return transfered data matrix.

Spon and series analysis will be done in other ways, not this class.

Input: good unit path, save path, stimset name
Return: Structured stim set.


'''
#%%
# from Cell_Class import Cell_Infos
import mat73
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
from joblib import dump, load
# from Struct_Funcs import Single_Recording_Site

#%%
class Single_Recording_Site(object):

    name = 'Info of a single recording site'

    def __init__(self,gn_path,stimset,
                 brain_areas=['MSB'],
                 onset=300,
                 offset=200,
                 used_on = np.arange(160,320),
                 save_train=False,
                 prepare_data = True
                 ):
        '''
        Save path need to be updated after load and save.
        '''

        ### acquire necessary infos
        self.gn_dic = mat73.loadmat(gn_path,verbose=False)
        self.site_name = self.gn_dic['meta_data']['ml_name'][:-5]
        self.stimset = stimset
        # self.savepath = ot.Join(savepath,self.site_name)
        self.cellnum = len(self.gn_dic['GoodUnitStrc']['unittype'])
        self.brain_areas = brain_areas
        self.onset = onset
        self.offset = offset
        self.img_size = int(self.gn_dic['meta_data']['img_size'])
        self.stim_info,_,_ = Load_Info(self.stimset)
        self.used_on = used_on

        #### Data prepare.
        if prepare_data:
            self.Data_Prepare(save_train=save_train)
            # after data transfer, del gn dict.
            del self.gn_dic


    def Data_Prepare(self,save_train):
        
        ## 1.save trains
        if save_train:
            print('Generating Raw Rasters...')
            self.raw_rasters,self.trail_index = Train_From_Goodunit(self.gn_dic,onset_time=self.onset)
            self.ISI_Extractor()
        ## 2.transfer psth. This is necessary.
        print('Generating PSTH...')
        self.raw_psth = PSTH_From_Goodunit(self.gn_dic,self.img_size,'dict')
        self.avr_psth = self.raw_psth.mean(1)
        ## 3. Noise ceiling, provide ceiling index.
        self.Noise_Ceiling()
        ## 4. Redplots
        self.raw_redplot = self.avr_psth[:,:,self.used_on].sum(-1)
        ## 5.calculate cell tunings.
        print('Calculating Cell Tuning Info...')
        self.FOB_Tuning_Calculator()
        ## 6. Final 
        print('Data processing done. Generating Overall info data frame...')
        self.Catable_Cell_Info_Generator()


    def Matrix_Bin_Lite(self,matrix,bin_tick=10):# Matrix in shape : N_CellxN_Timepoint
        n_bins = matrix.shape[1] // bin_tick
        binned_matrix = matrix[:,:n_bins*bin_tick].reshape(matrix.shape[0], n_bins, bin_tick).sum(2)
        return binned_matrix

        
    def Noise_Ceiling(self,method='all'): # keep an api for further possible alter.

        if method == 'all':
            self.ceiling_index = odd_end_ceiling(self.raw_psth,used_time=self.used_on)
        elif method =='fob':
            fob_parts =  self.stim_info[self.stim_info['Stim_Set'].str.contains('FOB', na=False)].index.tolist()
            self.ceiling_index = odd_end_ceiling(self.raw_psth[:,:,fob_parts,:],used_time=self.used_on)


        # return ok cell id and cell index.

    def FOB_Tuning_Calculator(self):
        self.Cell_FOB_Response = pd.DataFrame(index=range(100000000),columns = ['Cell','Response','Category'])# each frame will 
        self.Cell_FOB_DPrimes = pd.DataFrame(index=range(100000000),columns = ['Cell','D_Prime','Category'])

        # get fob info sets for data processing.

        fob_parts = self.stim_info[self.stim_info['Stim_Set'].str.contains('FOB', na=False)]
        if len(fob_parts)==0: # ifno avaliable fob
            print('FOB Method not supported or No FOB.')
            self.Cell_FOB_Response = self.Cell_FOB_Response.dropna(how='any')
            self.Cell_FOB_DPrimes = self.Cell_FOB_DPrimes.dropna(how='any')
            # raise ValueError('FOB Struct Not supported.')
            return # end this function.
        
        fob_style = fob_parts['Stim_Set'].iloc[0]
        fob_resps = self.raw_redplot[:,np.array(fob_parts.index)]# already cutted, so we can index from 0.

        # label fob, getting info of response.
        stim_cats = ['Face','Body','Object','Scene','Food'] # maybe I can just keep all categories.
        if 'STI150' in fob_style:
            stim_cats = ['Face','Body','Object','Scene','Food']
        elif 'FOB72' in fob_style:
            stim_cats = ['Face','Body','Object']
        else:
            raise ValueError('FOB Method unsupported.')
        
        counter=0
        for i in range(len(fob_parts)):
            c_cat = fob_parts.iloc[i,2]
            for j,c_stim in enumerate(stim_cats):
                if c_stim in c_cat:
                    for k in range(len(fob_resps)):
                        self.Cell_FOB_Response.loc[counter] = [k,fob_resps[k,i],c_stim]
                        counter+=1
        self.Cell_FOB_Response = self.Cell_FOB_Response.dropna(how='any')
        self.Cell_FOB_Response['Response']=self.Cell_FOB_Response.Response.astype('f8')# data type to float
        self.Cell_FOB_Response_avr = self.Cell_FOB_Response.groupby(['Cell','Category'])['Response'].mean().reset_index()
        #pivot = Cell_FOB_Response_avr.pivot(index='Cell', columns='Category', values='Response')

        # get d prime info of all cells.
        counter=0
        for i in range(len(fob_resps)): # cycle cells
            cc_response = self.Cell_FOB_Response[self.Cell_FOB_Response.Cell==i]
            for j,c_stim in enumerate(stim_cats):
                b_parts = cc_response[cc_response.Category==c_stim].Response.values
                a_parts = cc_response[cc_response.Category!=c_stim].Response.values
                cc_dp = np.nan_to_num(D_Prime(a_parts,b_parts))
                self.Cell_FOB_DPrimes.loc[counter] = [i,cc_dp,c_stim]
                counter+=1
        self.Cell_FOB_DPrimes = self.Cell_FOB_DPrimes.dropna(how='any')
        self.Cell_FOB_DPrimes['D_Prime'] = self.Cell_FOB_DPrimes.D_Prime.astype('f8')
        # pivot = Cell_FOB_DPrimes.pivot(index='Cell', columns='Category', values='D_Prime')


    def Catable_Cell_Info_Generator(self):
        
        self.Site_Info = pd.DataFrame(range(self.cellnum),index = range(self.cellnum),columns = ['Cell'])
        ac_locs = np.array(self.gn_dic['GoodUnitStrc']['spikepos']) # in shape x,y
        # extract cell info, including cell type and x-y location.
        ac_types = np.array(self.gn_dic['GoodUnitStrc']['unittype'])

        # columns = ['Best_Prefer','Best_D_Prime']
        self.Site_Info['Ceiling_Index']=self.ceiling_index
        self.Site_Info['Cell_yLoc']=ac_locs[:,1]
        self.Site_Info['Cell_Type']=ac_types
        
        # cycle cell for best d prime and 
        if len(self.Cell_FOB_DPrimes) != 0:
            best_dps = []
            best_dp_cats = []
            for i in range(self.cellnum):
                c_pref = self.Cell_FOB_DPrimes[self.Cell_FOB_DPrimes.Cell==i]
                best_dps.append(c_pref.D_Prime.max())
                best_dp_cats.append(c_pref.loc[c_pref.D_Prime.idxmax(),'Category'])
            self.Site_Info['Best_Prefer'] = best_dp_cats
            self.Site_Info['Best_D_Prime'] = best_dps
        else:
            print('FOB Not provided.')
            self.Site_Info['Best_Prefer'] = 'None'
            self.Site_Info['Best_D_Prime'] = 0

        # site info
        self.Site_Info['Site'] = self.site_name
        self.Site_Info['Brain_Area'] = [self.brain_areas for _ in range(self.cellnum)]

    def ISI_Extractor(self):
        ISI_infos,prev_ids,have_break = ML_trail_Cutter(self.trail_index)
        # counter=0
        self.ISI_Info_Frame = pd.DataFrame(index=range(len(ISI_infos)),columns=['Before_Stim','Breaked','Raster_Series'])
        for i,c_series in enumerate(ISI_infos):
            c_frame = self.raw_rasters[:,c_series]
            self.ISI_Info_Frame.loc[i,:] = [prev_ids[i],have_break[i],c_frame]

    def __len__(self):
        return len(self.raw_psth)

    def __getitem__(self,index):
        return self.avr_psth[index,:,:]
    

#%%

if __name__ == '__main__':

    # gn_path = r'E:\#Preprocessed_Data\GoodUnits\GoodUnit_251108_ZhuangZhuang_Metamer_Singlebubble_v251107_4540_g2_MSB.mat'
    gn_path = r'E:\#Preprocessed_Data\GoodUnits\GoodUnit_250829_ZhuangZhuang_Metamer_Pool4_C4321_Object_1k+FOB_g2_MSB_1072.mat'
    # save_path = r'E:\#Preprocessed_Data\SiteClass'
    stimset = 'Metamer1072'

    SRS = Single_Recording_Site(gn_path,stimset,['MSB'],300,200,True)
    # SRS.Data_Prepare
    dump(SRS, 'compressed_object.joblib', compress=8)  # compress=0-9, 数字越大压缩率越高

    # SRS.__dict__.keys()

