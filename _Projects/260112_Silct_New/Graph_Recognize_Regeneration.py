'''
识别图中物体，然后进行重新绘制。
用来生成物体相同但像素层面不一样的图片，这个模型使用千问2.6

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
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message
import requests
import dashscope
import os
from dashscope import ImageSynthesis

raw_silct_path = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\Gray_Norm_White_HD'
api_key = 'sk-输入api key'
savepath_raw = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Regeneration_1'


#%%

indices = [61]
all_names = ot.Get_File_Name(raw_silct_path,'.jpg')
all_names = [all_names[i] for i in indices]
# prompt 1 - HD graph only.
prompt=r"识别给定简笔画图中的物体，并生成写实风格的物体图片。物体在空白的背景上。"



#%% 以下为北京地域base_url，各地域的base_url不同
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"

message = Message(
    role="user",
    # 支持本地文件 如 "image": "file://umbrella1.png"
    content=[
        {
            "text": prompt
        },
        {
            "image": all_names[0]
        }
    ]
)
print("----sync call, please wait a moment----")
rsp = ImageGeneration.call(
        model='wan2.6-image',
        api_key=api_key,
        messages=[message],
        negative_prompt="",
        prompt_extend=True,
        watermark=False,
        n=1,
        enable_interleave=False,
        size="800*800"
    )

print(rsp)