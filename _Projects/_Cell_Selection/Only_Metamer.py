'''
This script will select good body cells in MSB.

NOTE only body cells, and only metamer selected. 

'''


#%%

from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
import copy
import warnings
import gc
warnings.filterwarnings("ignore")

msb_sites = ot.Get_File_Name(r'E:\#Preprocessed_Data\SiteClass\Metamers\MSB','.joblib')
save_path = r'E:\#Preprocessed_Data\Selected_Cells'
#%%
cell_counter = 0
for i,cloc in tqdm(enumerate(msb_sites)):
    print(cloc.split('\\')[-1])
    a = JL.load(cloc)
    c_info = a.Site_Info
    c_loc = a.site_name
    if ('ML' not in a.brain_areas) and ('MF' not in a.brain_areas):
    # if 'AL' not in a.brain_areas:
        print(f'Not Correct Area, ignore {cloc.split('\\')[-1]}.')
        del a
        gc.collect()
        continue
    ok_cells = c_info[c_info.Ceiling_Index>(0.3/1.7)]
    # body_cells = ok_cells[ok_cells.Best_Prefer=='Body']
    body_cells = ok_cells[ok_cells.Best_Prefer=='Face']
    # body_cells = np.array(ok_cells[ok_cells.Best_D_Prime>0.5].Cell)
    body_cells = np.array(body_cells[body_cells.Best_D_Prime>0.5].Cell)
    body_resps = a.avr_psth[body_cells,:,:]
    raw_rsp = a.avr_psth
    if a.stimset == 'Metamer1072':
        metamers = body_resps[:,72:,:]
    elif a.stimset == 'Metamer1300':
        metamers = body_resps[:,:1000,:]
    else:
        metamers = body_resps[:,300:1300,:]
    # generate concated response.
    try:# If no var, init.
        all_matemer_resp = np.concatenate([all_matemer_resp,metamers],axis=0)
    except:
        all_matemer_resp = copy.deepcopy(metamers)

    # generate cell infos.
    c_dps = a.Cell_FOB_DPrimes
    c_rsp = a.Cell_FOB_Response_avr
    c_dps = c_dps[c_dps['Cell'].isin(body_cells)] # only body-selected part
    c_rsp = c_rsp[c_rsp['Cell'].isin(body_cells)]

    # add loc.
    c_dps['Loc'] = c_loc
    c_rsp['Loc'] = c_loc

    try:
        all_response = pd.concat((all_response,c_rsp))
    except:
        all_response = copy.deepcopy(c_rsp)
    try:
        all_d_primes = pd.concat((all_d_primes,c_dps))
    except:
        all_d_primes = copy.deepcopy(c_dps)

    del a
    gc.collect()

all_d_primes['Cell_ID'] = all_d_primes.groupby(['Loc', 'Cell'], sort=False).ngroup()
all_response['Cell_ID'] = all_response.groupby(['Loc', 'Cell'], sort=False).ngroup()


#%% save all msb cells
np.savez_compressed(ot.Join(save_path,'ML_Cells_Metamer_Only.npz'),psth = all_matemer_resp,d_primes = all_d_primes,response = all_response)

print(set(all_d_primes.Loc))
#%% counts 
counter = 0
acs = all_d_primes[all_d_primes.Category=='Face'].reset_index(drop=True)
for i in range(len(acs)):
    cc_loc = acs.loc[i,'Loc']
    # if ('MD' in cc_loc) or ('Mao' in cc_loc):
    if 'Jian' in cc_loc:
        counter+=1
print(len(acs))
print(counter)


