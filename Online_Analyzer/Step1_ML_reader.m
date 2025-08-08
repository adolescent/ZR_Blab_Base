
%% 只有这部分需要操作，别的一般不用改

workpath = 'D:\#Data\Loc_Example\test_wp'; % tsv和bhv2文件所在的位置
npxpath = 'D:\#Data\Loc_Example\npx_root_decoy\NPX_MD241029_exp_g0';% 当前的npxrun



%% Load data and generate condition-binned rasters 

% clear
root_dir = 'D:\#Codes\ZR_Blab_Base\Online_Analyzer'; % util path
% cd(root_dir)
addpath(genpath('C:\Users\admin\AppData\Roaming\MathWorks\MATLAB Add-Ons\Apps\NIMHMonkeyLogic22')) % ml path
addpath(genpath(root_dir))
% align time series and generate successful time in ms.
Load_Data_function_Seperate(workpath,npxpath);


%% get valuable var from processed ML data.
meta_file = dir('processed/META*');
load(fullfile(pwd,'processed',meta_file(1).name));
meta_data = load(fullfile(pwd,'processed',meta_file(1).name));
ML_FILE = dir("processed\ML*");
trial_ML = load(fullfile('processed',ML_FILE(1).name)).trial_ML;
% PostProcess_function_raw(localizer_path);
% PostProcess_function(localizer_path);

%% save time of event to a csv 
trail_times = table(onset_time_ms.', trial_valid_idx.');
trail_times.Properties.VariableNames = {'Onset_Time', 'Trail_ID'};
writetable(trail_times, 'Onset_Times.csv');