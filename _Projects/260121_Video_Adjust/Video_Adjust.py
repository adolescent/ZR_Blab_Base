'''
把原来的视频改为1920*1080，并在左下角放白色方块。
'''

#%%
import ffmpeg
import OS_Tools as ot
import numpy as np
from tqdm import tqdm


#%%
raw_path = r'E:\#Stimsets\Face_Reconstruction\Raw_Video'
savepath = r'E:\#Stimsets\Face_Reconstruction\Resized_Video'
all_videos = ot.Get_File_Name(raw_path,'.avi')
all_videos.sort()
output_file = 'output.avi'
input_file = all_videos[0]


# 使用ffprobe获取输入视频的码率
def get_video_bitrate(input_file):
    try:
        # 获取视频信息
        probe = ffmpeg.probe(input_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if video_stream and 'bit_rate' in video_stream:
            bitrate = int(video_stream['bit_rate'])
            # 转换为Mbps格式的字符串
            bitrate_mbps = f"{bitrate // 1000}k"
            return bitrate_mbps
    except Exception as e:
        print(f"无法获取视频码率，使用默认值: {e}")
    
    # 如果无法获取码率，使用一个合理的默认值
    return "5000k"  # 5Mbps作为默认值

def Graph_Resolution_Change_Dialode(input_file,output_file):
    # 获取输入视频的码率
    input_bitrate = get_video_bitrate(input_file)
    # print(f"输入视频码率: {input_bitrate}")

    # 计算pad参数
    # 目标分辨率: 1920x1080
    # 原始分辨率: 1024x1024
    # 水平方向pad: (1920 - 1024) / 2 = 448
    # 垂直方向pad: (1080 - 1024) / 2 = 28

    # 使用pad滤镜将视频扩展到1920x1080，黑色边框，居中
    # 格式: pad=宽度:高度:左填充:上填充:颜色
    pad_filter = 'pad=1920:1080:448:28:black'

    # 添加白色方块
    # 由于视频已经扩展到1920x1080，原视频内容居中
    # 原视频左下角在整体画布中的坐标:
    # x = 448 (左填充)
    # y = 28 + 1024 - 50 = 1002 (上填充 + 原视频高度 - 方块高度)
    drawbox_filter = 'drawbox=x=448:y=1002:w=50:h=50:color=white@1.0:t=fill'

    # 组合滤镜链
    filter_chain = f'{pad_filter},{drawbox_filter}'

    try:
        # 处理视频
        stream = ffmpeg.input(input_file)
        
        # 应用滤镜链
        stream = ffmpeg.filter(stream, 'pad', 1920, 1080, 448, 28, color='black')
        stream = ffmpeg.filter(stream, 'drawbox', x=0, y=1010, width=70, height=70, color='white', thickness='fill')
        
        # 输出，保持原始码率
        stream = ffmpeg.output(
            stream,
            output_file,
            vcodec='mpeg4',  # 使用MPEG编码
            video_bitrate=input_bitrate,  # 使用输入视频的码率
            r=30,             # 保持30帧
            pix_fmt='yuv420p',
            **{'b:v': input_bitrate}  # 另一种设置码率的方式
        )
        
        # 运行ffmpeg
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        # print(f"视频处理完成！输出文件：{output_file}")
        # print(f"输出码率: {input_bitrate}")
        
    except ffmpeg.Error as e:
        print(f"处理视频时出错: {e.stderr.decode()}")
    except Exception as e:
        print(f"发生错误: {e}")

#%% Run parts
if __name__ == '__main__':
    
    raw_path = r'E:\#Stimsets\Face_Reconstruction\Raw_Video'
    savepath = r'E:\#Stimsets\Face_Reconstruction\Resized_Video'
    raw_names = ot.Get_File_Name(raw_path,'.avi')
    for i,c_filename in tqdm(enumerate(raw_names)):
        c_name = 'Emotion_'+c_filename.split('\\')[-1][:-4]+'.avi'
        c_savefile = ot.Join(savepath,c_name)
        Graph_Resolution_Change_Dialode(c_filename,c_savefile)