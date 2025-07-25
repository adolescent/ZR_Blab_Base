clear
root_dir = 'E:\ZR_Temp\Process_pipeline_2025April';
cd(root_dir)
addpath(genpath('C:\Users\admin\AppData\Roaming\MathWorks\MATLAB Add-Ons\Apps\NIMHMonkeyLogic22'))
addpath(genpath(root_dir))

% data_dir = 'J:\AxisRaw';
% data_sets = {'BigFOB_dataset','FFF_dataset','IMG2022_dataset','WordProject'};
% interested_path={};
% for dd = 1:length(data_sets)
%     all_ses = dir(fullfile(data_dir, data_sets{dd}));
%     for ss = 3:length(all_ses)
%         ses_dir = fullfile(data_dir, data_sets{dd}, all_ses(ss).name);
%         interested_path{end+1}=ses_dir;
%     end
% end
interested_path{1}='E:\ZR_Temp\_Data_Temp\JJ_250411_silct_MSB_mismatch';
% interested_path{2}='E:\MKB_project\MKB\JJ_MKI_0328';
% interested_path{3}='E:\MKB_project\MKB\ZZ_MKI_0328';
% interested_path{1}='E:\MKB_project\MKB\250325_ZZ';
%%
for path_now = 1:length(interested_path)
    close all
    Load_Data_function(interested_path{path_now});
    PostProcess_function_raw(interested_path{path_now});
    PostProcess_function(interested_path{path_now});
%     wm_dir = fullfile(interested_path{path_now},'processed','BC','RawWaveforms');
end