'''
这个脚本把原始图片切出来，然后做成HD化的。

使用千问api，大概需要80块。

NOTE 小心API！

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

raw_silct_path = r'E:\#Stimsets\Silct\silct_npx_1416'
api_key = 'sk-输入apikey'
savepath_raw = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle'

#%% 1 - 选取真实silct图片
import shutil

all_names = ot.Get_File_Name(raw_silct_path)[:1200]
doodles = all_names[::3]
for i in tqdm(range(len(doodles))):
    tar_path = ot.Join(savepath_raw,f'{i+10001}.jpg')
    sor_path = doodles[i]
    shutil.copy(sor_path, tar_path)
#%% ############################# HD ####################################
folder = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\Batch2'
tar = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\Batch2_HD'
all_names = ot.Get_File_Name(folder,'.jpg')
# prompt 1 - HD graph only.
prompt=r"去除图片中的噪点，提高图片质量。把图片中的全部白色变成127的灰色，使得整张图片是在灰色背景上的线条图。最终图片必须是只有灰色和黑色的二色图。其他部分不要修改。"
# prompt=r"提取图片中的线条图，生成一张127灰色的图片，把线条图绘制到新的图片上。保持线条图在原图和新图中的位置不改变。"

#%%
for i,c_img in tqdm(enumerate(all_names)):
    filename = c_img.split('\\')[-1][:-4]+'_HD.jpg'
    rsp = ImageSynthesis.call(api_key=api_key,
                                model="wan2.5-i2i-preview",
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
    save_filepath =ot.Join(tar,filename)
    with open(save_filepath, 'wb+') as f:
        f.write(requests.get(result.url).content)
    del rsp



#%%
if __name__ == '__main__':
    test_img = '0010.jpg'
    base_image_url = "file://"+f"./{test_img}"  # Windows
    prompt=r'Remove noise points of this graph, improve graphic quality, fill white back ground of this graph into 127 gray color, including white background inside the image, making the image line arts layed on gray background, only black lines and gray background, no other color. Do not change other parts of the image.'
    # prompt=r"去除图片中的噪点，提高图片质量，并把白色的部分填充为127的灰色，包括图片内部的方格，使得整张图片是在灰色背景上的线条。其他部分不要修改。"

    rsp = ImageSynthesis.call(api_key=api_key,
                          model="wan2.5-i2i-preview",
                          prompt=prompt,
                          images=[base_image_url],
                          negative_prompt="",
                          n=1,
                          size="800*800",
                          prompt_extend=True,
                          watermark=False,
                          #seed=114514
                          )
    counter=0
    for result in rsp.output.results:
        # file_name = str(counter)+PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
        file_name = f'{counter}_HD.png'
        with open('./%s' % file_name, 'wb+') as f:
            f.write(requests.get(result.url).content)
        counter+=1
    

