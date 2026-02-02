'''
增加了resize的功能，resize成600*600，大概15°视野。


'''


#%%
import ffmpeg
import OS_Tools as ot
import numpy as np
from tqdm import tqdm


raw_path = r'E:\#Stimsets\Face_Reconstruction\Raw_Video'
savepath = r'E:\#Stimsets\Face_Reconstruction\Resized_Video'
all_videos = ot.Get_File_Name(raw_path,'.avi')
all_videos.sort()
output_file = 'output.avi'
input_file = all_videos[0]

#%% 
# 获取输入视频信息
def Generate_Video(input_file,output_file):
    probe = ffmpeg.probe(input_file)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')

    # 获取原始编码器和码率
    codec_name = video_info['codec_name']
    bitrate = video_info.get('bit_rate', '2000k')  # 如果没有码率信息，使用默认值2Mbps

    # 计算pad的偏移量，使600x600视频在1918x1078中居中
    pad_x = (1918 - 600) // 2
    pad_y = (1078 - 600) // 2

    # 构建ffmpeg命令链
    (
        ffmpeg
        .input(input_file)
        .filter('scale', 600, 600)  # 缩放为600x600
        .filter('pad', 1918, 1078, pad_x, pad_y, color='black')  # pad为1918x1078，背景黑色
        .filter('drawbox', x=0, y=1018, width=70, height=70, color='white', t='fill')  # 左下角50x50白色方块
        .output(
            output_file,
            **{'c:v': codec_name,  # 使用原视频编码器
            'b:v': bitrate,     # 保持原码率
            'r': 30,            # 保持30帧
            'c:a': 'copy'}      # 音频直接复制
        )
        .overwrite_output()
        .run()
    )

    # print(f"视频处理完成！输出文件: {output_file}")
    # print(f"使用的编码器: {codec_name}")
    # print(f"使用的码率: {bitrate}")

#%%
if __name__ == "__main__":
    raw_path = r'E:\#Stimsets\Face_Reconstruction\Raw_Video'
    savepath = r'E:\#Stimsets\Face_Reconstruction\Resized_Video'
    raw_names = ot.Get_File_Name(raw_path,'.avi')
    for i,c_filename in tqdm(enumerate(raw_names)):
        c_name = 'Emotion_'+c_filename.split('\\')[-1][:-4]+'.avi'
        c_savefile = ot.Join(savepath,c_name)
        Generate_Video(c_filename,c_savefile)

        