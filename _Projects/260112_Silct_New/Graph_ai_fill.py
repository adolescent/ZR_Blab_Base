'''
使用生成式AI对图片进行补完，生成三种不同的变种。


'''


#%%
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
from tqdm import tqdm
import base64
import mimetypes
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import dashscope
import requests
from dashscope import ImageSynthesis

raw_silct_path = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\Gray_Norm_White_HD'
api_key = 'sk-输入api key'
savepath_raw = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Regeneration_1'

#%%
# indices = [23,26,31,56,57,59,64,66,67,72,77,83,87,89,91,92,96,101,102,105,106,108,114,117,125,126,128,129,133,138,139,141,143,147,150,151,154,162,172,174,179,181,183,186,191,198]
indices = [61]
all_names = ot.Get_File_Name(raw_silct_path,'.jpg')
all_names = [all_names[i] for i in indices]
# prompt 1 - HD graph only.
prompt=r"根据给定的简笔画，生成写实风格的物体图片。不要增加背景。物体的位置在像素上和原有的简笔画在像素上完全对应。"
# prompt = r"识别图中物体，并生成写实风格的物体。不要增加背景。像素不要与简笔画对应。"
# 这是一个镂空金属丝编织的鸟笼。
#%% 运行部分
for i,c_img in tqdm(enumerate(all_names)):
    filename = c_img.split('\\')[-1][:-4]+'_HD.jpg'
    rsp = ImageSynthesis.call(api_key=api_key,
                                model="wan2.5-i2i-preview",
                                # model = "wanx-sketch-to-image-lite",
                                prompt=prompt,
                                images=[c_img],
                                negative_prompt="",
                                n=1,
                                size="800*800",
                                prompt_extend=True,
                                watermark=False,
                                #seed=114514
                                )
    # save graph into target folder.
    result=rsp.output.results[0] # 多张图的话这里需要修改
    save_filepath =ot.Join(savepath_raw,filename)
    with open(save_filepath, 'wb+') as f:
        f.write(requests.get(result.url).content)
    del rsp

