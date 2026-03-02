'''

Combine stim of anagram types.


'''


#%%
import OS_Tools as ot
from PIL import Image
from tqdm import tqdm
import numpy as np

savepath = r'E:\#Stimsets\Silct_Localizer\Doodle_Localizer240'

counter=1

def resize_image(input_path, output_path, new_size=(400, 400)):
    """
    调整图片尺寸并保存
    
    Parameters:
    input_path: 输入图片路径
    output_path: 输出图片路径
    new_size: 新尺寸，默认为(400, 400)
    """
    with Image.open(input_path) as img:
        # 获取原图尺寸
        original_size = img.size
        # print(f"原图尺寸: {original_size}")
        
        # 调整图片尺寸
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
        resized_img.save(output_path)


#%% Batch 1
body_raw = ot.Get_File_Name(r'E:\#Stimsets\Silct_Localizer\body_real')
face_raw = ot.Get_File_Name(r'E:\#Stimsets\Silct_Localizer\face_real')
spiky_raw = ot.Get_File_Name(r'E:\#Stimsets\Silct_Localizer\spiky_real')
stubby_raw = ot.Get_File_Name(r'E:\#Stimsets\Silct_Localizer\stubby_real')
body_doodle = ot.Get_File_Name(r'E:\#Stimsets\Silct_Localizer\body_doodle','.png')
face_doodle = ot.Get_File_Name(r'E:\#Stimsets\Silct_Localizer\face_doodle','.png')
spiky_doodle = ot.Get_File_Name(r'E:\#Stimsets\Silct_Localizer\spiky_doodle','.png')
stubby_doodle = ot.Get_File_Name(r'E:\#Stimsets\Silct_Localizer\stubby_doodle','.png')



#%% resize graph and rename.
counter = 1
for j,c_set in enumerate([body_raw,face_raw,spiky_raw,stubby_raw,body_doodle,face_doodle,spiky_doodle,stubby_doodle]):
    for i,c_img in enumerate(c_set):

        c_name_new_a = str(10000+counter)+'.png'
        save_filename_a = ot.Join(savepath,c_name_new_a)
        resize_image(c_img,save_filename_a)
        counter += 1


#%%
############################ GENERATE FOB INFO and Python-style INFO ##############################

import csv

# 定义输出文件名
output_file = 'doodleLOC_info.tsv'

# 定义数据行数
num_rows = 240

# 准备数据（包含标题行）
data = []
# 添加标题
data.append(['Index', 'FileName', 'Category', 'FOB'])

# 生成数据行
for i in range(1, num_rows + 1):
    file_name = f"{10000 + i}.png"  # 1001.jpg, 1002.jpg, ...
    c_cat = (i-1)//30
    all_labels = ['Body','Face','Spiky','Stubby','Body_s','Face_s','Spiky_s','Stubby_s']
    row = [i, file_name, 'DoodleLoc',all_labels[c_cat]]
    data.append(row)

# 写入CSV文件（使用制表符分隔）
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(data)

print(f"CSV文件已生成：{output_file}")
#%%
############################### PYTHON STYLE INFO ######################################
import pandas as pd

# 定义输出文件名
# output_file = 'DoodleLOC.tsv'

data = []
# 添加标题
data.append(['FileName','Category','Stim_Set','Object'])

# 定义数据行数
num_rows = 240

# 生成数据行
for i in range(1, num_rows + 1):
    file_name = f"{10000 + i}.png"  # 1001.jpg, 1002.jpg, ...
    c_cat = (i-1)//30
    all_labels = ['Body','Face','Spiky','Stubby','Body_s','Face_s','Spiky_s','Stubby_s']
    row = [ file_name, all_labels[c_cat],'FOB_DoodleLOC',-1]
    data.append(row)
    
# 写入CSV文件（使用制表符分隔）
df = pd.DataFrame(data[1:], columns=data[0])
df.to_csv( 'DoodleLOC.tsv', sep='\t', index=True)