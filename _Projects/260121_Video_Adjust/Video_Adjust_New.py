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


def get_video_info(input_file):
    """获取输入视频的详细信息"""
    try:
        # 使用ffprobe获取视频信息
        probe = ffmpeg.probe(input_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if not video_stream:
            raise Exception("未找到视频流")
        
        # 提取视频信息
        info = {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'codec_name': video_stream.get('codec_name', 'mpeg4'),  # AVI通常使用mpeg4
            'bit_rate': int(video_stream.get('bit_rate', 5000000)) if 'bit_rate' in video_stream else 5000000,
            'r_frame_rate': video_stream.get('r_frame_rate', '30/1'),
            'pix_fmt': video_stream.get('pix_fmt', 'yuv420p'),
        }
        
        # 获取编码器参数
        if 'codec_tag_string' in video_stream:
            info['codec_tag'] = video_stream['codec_tag_string']
        
        return info
        
    except Exception as e:
        print(f"获取视频信息时出错: {e}")
        # 返回默认值
        return {
            'width': 1024,
            'height': 1024,
            'codec_name': 'mpeg4',
            'bit_rate': 5000000,
            'r_frame_rate': '30/1',
            'pix_fmt': 'yuv420p',
        }

def main(input_file,output_file):
    # 获取输入视频信息
    video_info = get_video_info(input_file)
    # print("输入视频信息:")
    # print(f"  分辨率: {video_info['width']}x{video_info['height']}")
    # print(f"  编码器: {video_info['codec_name']}")
    # print(f"  码率: {video_info['bit_rate'] // 1000} kbps")
    # print(f"  帧率: {video_info['r_frame_rate']}")
    # print(f"  像素格式: {video_info['pix_fmt']}")
    
    # 目标分辨率
    target_width = 1918
    target_height = 1078
    
    # 计算pad参数
    # 水平方向pad: (1918 - 1024) / 2 = 447
    # 垂直方向pad: (1078 - 1024) / 2 = 27
    pad_left = (target_width - video_info['width']) // 2
    pad_top = (target_height - video_info['height']) // 2
    
    # print(f"\nPad参数:")
    # print(f"  目标分辨率: {target_width}x{target_height}")
    # print(f"  左/右填充: {pad_left} 像素")
    # print(f"  上/下填充: {pad_top} 像素")
    
    # 白色方块位置
    # 原视频左下角在画布中的坐标:
    # x = pad_left
    # y = pad_top + video_info['height'] - 50
    # white_box_x = pad_left
    # white_box_y = pad_top + video_info['height'] - 50
    
    # print(f"\n白色方块位置:")
    # print(f"  x: {white_box_x}, y: {white_box_y}, 大小: 50x50")
    
    # 构建ffmpeg命令
    try:
        # 创建输入流
        input_stream = ffmpeg.input(input_file)
        
        # 应用pad滤镜（不缩放，只添加黑色边框）
        stream = ffmpeg.filter(
            input_stream, 
            'pad', 
            width=target_width, 
            height=target_height, 
            x=pad_left, 
            y=pad_top, 
            color='black'
        )
        
        # 添加白色方块
        stream = ffmpeg.filter(
            stream,
            'drawbox',
            x=0,
            y=1008,
            width=70,
            height=70,
            color='white',
            thickness='fill'
        )
        
        # 准备输出参数
        output_args = {
            'vcodec': video_info['codec_name'],  # 使用相同的编码器
            'b:v': f"{video_info['bit_rate']}",  # 使用相同的码率
            'r': video_info['r_frame_rate'],     # 保持相同帧率
            'pix_fmt': video_info['pix_fmt'],    # 保持相同像素格式
        }
        
        # 添加额外的编码器参数（如果可用）
        if 'codec_tag' in video_info:
            output_args['vtag'] = video_info['codec_tag']
        
        # 输出文件
        output_stream = ffmpeg.output(stream, output_file, **output_args)
        
        # 运行ffmpeg
        # print(f"\n开始处理视频...")
        ffmpeg.run(output_stream, overwrite_output=True, quiet=False)
        
        # print(f"\n视频处理完成！")
        # print(f"输出文件: {output_file}")
        # print(f"输出分辨率: {target_width}x{target_height}")
        # print(f"输出编码器: {video_info['codec_name']}")
        # print(f"输出码率: {video_info['bit_rate'] // 1000} kbps")
        
    except ffmpeg.Error as e:
        print(f"FFmpeg处理错误: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        print(f"处理视频时发生错误: {str(e)}")
#%%
if __name__ == "__main__":
    raw_path = r'E:\#Stimsets\Face_Reconstruction\Raw_Video'
    savepath = r'E:\#Stimsets\Face_Reconstruction\Resized_Video'
    raw_names = ot.Get_File_Name(raw_path,'.avi')
    for i,c_filename in tqdm(enumerate(raw_names)):
        c_name = 'Emotion_'+c_filename.split('\\')[-1][:-4]+'.avi'
        c_savefile = ot.Join(savepath,c_name)
        main(c_filename,c_savefile)