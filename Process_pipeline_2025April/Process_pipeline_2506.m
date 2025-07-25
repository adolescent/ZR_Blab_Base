clear
close all
root_dir = 'E:\ZR_Temp\Process_pipeline_2025April';
cd(root_dir)
addpath(genpath('C:\Users\admin\AppData\Roaming\MathWorks\MATLAB Add-Ons\Apps\NIMHMonkeyLogic22'))
addpath(genpath(root_dir))
data_dir = 'E:';
data_sets = {'ZR_Temp\_Data_Temp\JJ_250411_silct_MSB'};
interested_path={};
interested_path{1}='E:ZR_Temp\_Data_Temp\JJ_250411_silct_MSB';
for dd = 1:length(data_sets)
    all_ses = dir(fullfile(data_dir, data_sets{dd}));
    for ss = 3:length(all_ses)
        ses_dir = fullfile(data_dir, data_sets{dd}, all_ses(ss).name);
        interested_path{end+1}=ses_dir;
    end
end

% data_dir = 'K:\';
% data_sets = {'AxisProject'};
% for dd = 1:length(data_sets)
%     all_ses = dir(fullfile(data_dir, data_sets{dd}));
%     for ss = 3:length(all_ses)
%         ses_dir = fullfile(data_dir, data_sets{dd}, all_ses(ss).name);
%         interested_path{end+1}=ses_dir;
%     end
% end
% 
% interested_path = interested_path';


%%
for path_now = 32
    Load_Data_function(interested_path{path_now});
    PostProcess_function_raw(interested_path{path_now});
    PostProcess_function(interested_path{path_now});

    path_here = dir(fullfile(interested_path{path_now},'processed','GoodUnit_*'));
    path_here = fullfile(interested_path{path_now},'processed',path_here(1).name);
    copyfile(path_here,'K:\SummaryAxis')
end