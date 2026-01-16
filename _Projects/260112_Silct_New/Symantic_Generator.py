'''
使用ai，根据语义生成意思相同，但细节完全不同的图案。

'''


#%%

import OS_Tools as ot
from tqdm import tqdm
import pandas as pd
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis
import os
import dashscope

savepath_raw = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Regeneration_2'
api_key = r'sk-输入api'
start = 0
names = pd.read_csv(r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\Images.csv',encoding='gbk').iloc[start:,:]

#%% ai生成部分



# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


# c_obj = names.iloc[34,1]
# prompt = f"在127的灰色背景下生成以下物体：{c_obj},只生成物体，不增加任何其它内容。"

for i in tqdm(range(len(names))):
    c_obj = names.iloc[i,1]
    filename = f'{i+start+10001}_HD.jpg'
    # print('----同步调用，请等待任务执行----')
    prompt = f"在127的灰色背景下生成以下物体：{c_obj},只生成物体，不增加任何其它内容。"
    rsp = ImageSynthesis.call(api_key=api_key,
                            # model="wanx2.1-t2i-turbo", # 当前仅qwen-image-plus、qwen-image模型支持异步接口
                            model="wanx2.1-t2i-plus",
                            prompt=prompt,
                            negative_prompt=" ",
                            n=1,
                            size='800*800',
                            prompt_extend=True,
                            watermark=False)
    # print(f'response: {rsp}')
    result=rsp.output.results[0] # 多张图的话这里需要修改
    save_filepath =ot.Join(savepath_raw,filename)
    with open(save_filepath, 'wb+') as f:
        f.write(requests.get(result.url).content)
    del rsp


#%% ## 单张图微调
# prompt = f"生成一个外卡钳（Calipers），真实图片风格。背景是灰色的纯色背景"
prompt= '生成一座双悬窗，真实图片风格。背景是灰色的纯色背景'
rsp = ImageSynthesis.call(api_key=api_key,
                        # model="wanx2.1-t2i-turbo", # 当前仅qwen-image-plus、qwen-image模型支持异步接口
                        model="wanx2.1-t2i-plus",
                        prompt=prompt,
                        negative_prompt=" ",
                        n=1,
                        size='800*800',
                        prompt_extend=True,
                        watermark=False)
result=rsp.output.results[0]
save_filepath = '10396_HD.jpg'
with open(save_filepath, 'wb+') as f:
    f.write(requests.get(result.url).content)
del rsp


