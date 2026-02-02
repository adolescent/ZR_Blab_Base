'''
把FOB变成普通的视频，
'''
#%%
import ffmpeg
import OS_Tools as ot
import numpy as np
from tqdm import tqdm
import os

# input_image = fobs[0]

def create_video_from_image(input_image,output_video = 'output.avi'):
    # 输入参数
    
    # 视频参数
    target_width = 1918
    target_height = 1078
    fps = 30
    duration = 2  # 秒
    bitrate = '2M'  # 2Mbps
    
    # 检查输入图片是否存在
    if not os.path.exists(input_image):
        print(f"错误：图片文件 '{input_image}' 不存在！")
        return
    
    # 计算居中位置
    # 图片分辨率 (假设是400x400，但我们会从输入获取实际分辨率)
    probe = ffmpeg.probe(input_image)
    image_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    
    if image_stream:
        img_width = int(image_stream['width'])
        img_height = int(image_stream['height'])
    else:
        # 如果无法获取图片信息，使用默认值
        img_width = 400
        img_height = 400
    
    # 计算居中位置
    x_offset = (target_width - img_width) // 2
    y_offset = (target_height - img_height) // 2
    
    try:
        # 构建ffmpeg命令
        # 使用loop滤镜重复图片，pad滤镜添加灰色背景
        stream = ffmpeg.input(input_image, loop=1, framerate=fps, t=duration)
        # 应用滤镜：将图片居中放置，用灰色填充周围
        # color=0x7F7F7F 表示 RGB(127, 127, 127)
        stream = stream.filter('pad', 
                               width=target_width, 
                               height=target_height, 
                               x=x_offset, 
                               y=y_offset, 
                               color='0x7F7F7F')
        # 增加白色二极管标志
        stream = ffmpeg.filter(stream, 'drawbox', x=0, y=1010, width=70, height=70, color='white', thickness='fill')
        # 输出设置
        stream = ffmpeg.output(stream, output_video, 
                               video_bitrate=bitrate, 
                               r=fps, 
                               pix_fmt='yuv420p')
        
        # 执行命令
        ffmpeg.run(stream, overwrite_output=True)
        
        # print(f"视频已成功创建：{output_video}")
        # print(f"参数：{target_width}x{target_height}, {fps}fps, {bitrate}bps, {duration}秒")
        
    except ffmpeg.Error as e:
        print(f"FFmpeg错误：{e.stderr.decode() if e.stderr else '未知错误'}")
    except Exception as e:
        print(f"发生错误：{str(e)}")
#%%
if __name__ == '__main__':
    
    fob_path = r'E:\#Stimsets\Face_Reconstruction\fob_raw'
    savepath = r'E:\#Stimsets\Face_Reconstruction\fob_videos'
    fobs = ot.Get_File_Name(fob_path,'.png')
    for i,c_fob in tqdm(enumerate(fobs)):
        c_name = 'FOB_'+c_fob.split('\\')[-1][:-4]+'.avi'
        c_savefile = ot.Join(savepath,c_name)
        create_video_from_image(c_fob,c_savefile)