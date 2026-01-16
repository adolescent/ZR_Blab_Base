'''
对简笔画图进行边界识别，reshape和boulder-silct处理
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
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


#%%


def extract_silhouette(image_path, output_contour='contour.png', output_silhouette='silhouette.png',min_contour_area=100):
    """
    提取物体轮廓并生成剪影图
    
    参数:
    image_path: 输入图片路径
    output_contour: 轮廓图保存路径
    output_silhouette: 剪影图保存路径
    """
    
    
    
    ""
    
    # 1. 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    
    # 2. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. 高斯模糊，减少噪点
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. 使用Canny边缘检测（更简洁的边缘）
    edges = cv2.Canny(blurred, 30, 100)
    
    # 5. 形态学操作，连接断开的边缘
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 6. 查找所有轮廓
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 7. 过滤掉小面积的轮廓（噪点）
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # 8. 如果找到多个轮廓，选择面积最大的一个
    if filtered_contours:
        main_contour = max(filtered_contours, key=cv2.contourArea)
        contours = [main_contour]
    

    # 6. 创建纯灰背景
    white_background = np.ones_like(gray) * 127
    
    # 7. 绘制轮廓图（黑色轮廓在灰色背景上）
    contour_image = white_background.copy()
    cv2.drawContours(contour_image, contours, -1, (0,0,0), 4)
    
    # # 8. 生成剪影图（黑色物体在灰色背景上）
    ## 剪影效果不好，手动调整后再改。
    # silhouette = white_background.copy()
    # cv2.drawContours(silhouette, contours, -1, (0,0,0), -1)  # -1表示填充
    
    # 9. 保存结果
    cv2.imwrite(output_contour, contour_image)

    return contour_image

#%% 使用方法
if __name__ == "__main__":

    
    all_doodle_path = ot.Get_File_Name(r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Real_1')
    boulder_savepath = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Real_1_Boulders_Raw'
    silct_savepath = r'N.A.'


    for i,image_path in tqdm(enumerate(all_doodle_path)):
        # 替换为你的图片路径
        c_filename = image_path.split('\\')[-1]
        counter_filename = ot.Join(boulder_savepath,c_filename)
        silct_filename = ot.Join(silct_savepath,c_filename)
    
        # 提取轮廓和剪影
        contour_img = extract_silhouette(
            image_path,
            output_contour=counter_filename,
            output_silhouette=silct_filename
        )