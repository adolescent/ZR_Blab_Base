'''
Syn graph into batches,

'''

#%%

from torchvision import transforms
from PIL import Image
import os
import Common_Functions.OS_Tools as ot
from PIL import Image, ImageStat
import numpy as np
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

all_img_name = ot.Get_File_Name(r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\raw','.jpg')
wp=r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\raw'
savepath = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\raw\cropped'

for i,c_graph in tqdm(enumerate(all_img_name)):

    filename = c_graph.split('\\')[-1]
    img = Image.open(c_graph)

    # 定义转换流程
    transform = transforms.Compose([
        # 第一步：中心裁剪为正方形（使用原始尺寸的最小边长）
        transforms.CenterCrop(min(img.size)),  # 关键步骤：保持原始比例裁剪
        
        # 第二步：缩放到400x400
        transforms.Resize((400, 400)),  # 双线性插值保持质量
        
        # 第三步：转换为张量（可选）
        transforms.ToTensor()
    ])
    processed_img = transform(img)
    processed_img_pil = transforms.ToPILImage()(processed_img)
    processed_img_pil.save(ot.Join(savepath,filename))


#%%
# 加载图片


processed_img = transform(img)
processed_img_pil = transforms.ToPILImage()(processed_img)


# savepath = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\raw\cropped'
processed_img_pil.save(ot.Join(savepath,filename+'raw.jpg'))

processed_img_pil

#%%
###################### SINGLE IMG Adjustment
'''
Codes below will adjust single graph.
'''

c_img_path = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\raw\Texture8_pomegranate.jpg'
filename = c_img_path.split('\\')[-1]
img = Image.open(c_img_path)
# 定义转换流程
transform = transforms.Compose([
    # 第一步：中心裁剪为正方形（使用原始尺寸的最小边长）
    transforms.CenterCrop(min(img.size)),  # 关键步骤：保持原始比例裁剪
    # 第二步：缩放到400x400
    transforms.Resize((400, 400)),  # 双线性插值保持质量
    
    # 第三步：转换为张量（可选）
    transforms.ToTensor()
])
processed_img = transform(img)
processed_img_pil = transforms.ToPILImage()(processed_img)
processed_img_pil.save(ot.Join(savepath,filename))
processed_img_pil


#%%




