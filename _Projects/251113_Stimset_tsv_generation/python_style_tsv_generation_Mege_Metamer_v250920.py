'''
This script will generate tsv info for stimsets we used here.

tsv in sequence:
ID,FileName,Stim_Type,Category,Raw_Graph

Category is divided by '_', e.g. P4_C1

'''
#%%

import OS_Tools as ot
import csv
import pandas as pd
import matplotlib.pyplot as plt



#%%
############################ P1-Mega_Metamer_v251104 #############################

# load ML tsv file
filename = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Metamer_Cut_v251011','.tsv')[0]
stim_infos = pd.read_csv(filename, sep='\t')

# add columns for annotate.
stim_infos['Stim_Set']='Default' # stim set of given stim
stim_infos['Category']='Default' # category of stimtype,sepereted by '_'
stim_infos['Object']=-1 # object of given stim, 1-40 as input data sets, and -1 for fob.

#%% generate fob 150*2 parts
# fob parts
stim_sets = []
categories = []
objects = []
for i in range(2): # 2 repeats
    for i in range(15):
        categories.append('Face')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Body')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Object')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Scene')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Food')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Face_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Body_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Object_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Scene_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)
    for i in range(15):
        categories.append('Food_g')
        stim_sets.append('FOB_STI150')
        objects.append(-1)

#%% 1-series, ordinary metamer parts

for i in range(5): # cycle repeats
    for j in range(5): # cycle constrains
        for k in range(40): # cycle objects
            stim_sets.append('Metamer')
            objects.append(k+1) # 1-40
            if j == 0:
                if k<20:
                    categories.append('Raw_Raw_Ani')
                else:
                    categories.append('Raw_Raw_Inani')
            elif j==1:
                if k<20:
                    categories.append('P4_C4_Ani')
                else:
                    categories.append('P4_C4_Inani')
            elif j==2:
                if k<20:
                    categories.append('P4_C3_Ani')
                else:
                    categories.append('P4_C3_Inani')
            elif j==3:
                if k<20:
                    categories.append('P4_C2_Ani')
                else:
                    categories.append('P4_C2_Inani')
            elif j==4:
                if k<20:
                    categories.append('P4_C1_Ani')
                else:
                    categories.append('P4_C1_Inani')


#%% 3-series, silct and boulder
for i in range(20):
    categories.append('Boulder')
    stim_sets.append('Boulder_Silct')
    objects.append(i+1)

for i in range(20):
    categories.append('Silct')
    stim_sets.append('Boulder_Silct')
    objects.append(i+1)


#%% 4-series, octo cuts, you might need mask combined for processing.
octo_methods = ['L1_1','L1_2','L1_3','L1_4','L1_5','L1_6','L1_7','L1_8','L2_12','L2_34','L2_56','L2_78','L2_15','L2_26','L2_37','L2_48','L4_1357','L4_2468','L4_1234','L4_5678','L4_3456','L4_1278','L6_345678','L6_125678','L6_123478','L6_123456','L7_2345678','L7_1345678','L7_1245678','L7_1235678','L7_1234678','L7_1234578','L7_1234568','L7_1234567'] # L meaning leave 1 parts, number below indicate the id of parts 

for i in range(34):
    for j in range(20):
        stim_sets.append('Cut_Octo')
        categories.append(octo_methods[i])
        objects.append(j+1)

#%% 5-series, N cuts, masks might be required.
n_methods = ['L1_1','L1_2','L1_3','L1_4','L1_5','L1_6','L1_7','L1_8','L1_9',
             'L3_123','L3_456','L3_789','L3_147','L3_258','L3_369','L3_159','L3_357',
             'L4_1359','L4_2468','L5_24568','L5_13579',
             'L6_123456','L6_123789','L6_456789','L6_235689','L6_134679','L6_124578','L6_234678','L6_124689',
             'L8_23456789','L8_13456789','L8_12456789','L8_12356789','L8_12346789','L8_12345789','L8_12345689','L8_12345679','L8_12345678'] 


# L meaning leave 1 parts, number below indicate the id of parts 

for i in range(38):
    for j in range(20):
        stim_sets.append('Cut_Ninegrid')
        categories.append(n_methods[i])
        objects.append(j+1)

#%% At last, combine all this into tsv file.
stim_infos['Category'] = categories
stim_infos['Stim_Set'] = stim_sets
stim_infos['Object'] = objects

stim_infos.to_csv('Metamer_Cut_v251011.tsv', sep='\t', index=True)
#%% 
''' 
For occulution and mask, we provide a different way of getting mask of each stim. including the mask center and full mask of given stim. Load it in if required.

'''
## occu mask already saved.
# occu_masks = ot.Load_Variable(r'E:\#Coding_traces\Stim_Generator_bubble_single_LC\Mask_Infos.pkl')

## Octo and N cut need generate as required.

import numpy as np
from PIL import Image, ImageDraw

def octo_mask(image_size, selected_parts):
    """
    创建米字切割的mask
    
    参数:
    image_size: 图片尺寸 (width, height)
    selected_parts: 要保留的部分列表，如 [1,2], [1,3,5], [1,6] 等
    
    返回:
    mask: 二值mask,选中的部分为1,其余为0
    """
    width, height = image_size
    mask = Image.new('L', (width, height), 0)  # 创建全黑mask
    draw = ImageDraw.Draw(mask)
    center_x, center_y = width // 2, height // 2
    
    # 创建一个足够大的半径来覆盖整个图像
    radius = int(np.sqrt(center_x**2 + center_y**2)) * 2
    
    # 绘制选中的扇形区域
    for part in selected_parts:
        if 1 <= part <= 8:
            start_angle = (part - 1) * 45
            end_angle = part * 45
            
            # 绘制扇形
            draw.pieslice([center_x - radius, center_y - radius, 
                          center_x + radius, center_y + radius],
                         start_angle, end_angle, fill=255)
    
    return mask

def n_mask(image_size, selected_parts):
    width, height = image_size
    mask = Image.new('L', (width, height), 0)  # 创建全黑mask
    draw = ImageDraw.Draw(mask)
    
    # 计算每个小块的宽度和高度
    block_width = width // 3
    block_height = height // 3
    
    # 九宫格编号（从左到右，从上到下）：
    # 1 2 3
    # 4 5 6
    # 7 8 9
    
    # 定义每个块的位置
    blocks = {
        1: (0, 0, block_width, block_height),
        2: (block_width, 0, 2 * block_width, block_height),
        3: (2 * block_width, 0, width, block_height),
        4: (0, block_height, block_width, 2 * block_height),
        5: (block_width, block_height, 2 * block_width, 2 * block_height),
        6: (2 * block_width, block_height, width, 2 * block_height),
        7: (0, 2 * block_height, block_width, height),
        8: (block_width, 2 * block_height, 2 * block_width, height),
        9: (2 * block_width, 2 * block_height, width, height)
    }
    
    # 绘制选中的块
    for part in selected_parts:
        if 1 <= part <= 9:
            x0, y0, x1, y1 = blocks[part]
            draw.rectangle([x0, y0, x1, y1], fill=255)
    
    return mask

# generate masks of all img.
mega_masks = np.zeros(shape = (len(stim_infos),400,400),dtype='bool')
mega_masks[:1340,:,:] = 1
#%% generate octo and n mask classes.
from tqdm import tqdm
octo_n_masks = []
image_size=(400,400)
# apply different cut methods to imgs, save them into different folder.
octo_paras = [[1],[2],[3],[4],[5],[6],[7],[8],#8
              [1,2],[3,4],[5,6],[7,8],[1,5],[2,6],[3,7],[4,8],#8
              [1,3,5,7],[2,4,6,8],[1,2,3,4],[5,6,7,8],[3,4,5,6],[1,2,7,8],#6
              [3,4,5,6,7,8],[1,2,5,6,7,8],[1,2,3,4,7,8],[1,2,3,4,5,6],#4
              [2,3,4,5,6,7,8],[1,3,4,5,6,7,8],[1,2,4,5,6,7,8],[1,2,3,5,6,7,8],[1,2,3,4,6,7,8],[1,2,3,4,5,7,8],[1,2,3,4,5,6,8],[1,2,3,4,5,6,7]#8
              ]

for j,c_octo_para in tqdm(enumerate(octo_paras)):
    for i in range(20):
        c_octo_mask = octo_mask(image_size,c_octo_para)
        octo_n_masks.append(c_octo_mask)

# and n cuts
n_paras = [[1],[2],[3],[4],[5],[6],[7],[8],[9],#One-9
            [1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9],[1,5,9],[3,5,7],#Three-8
            [1,3,7,9],[2,4,6,8],[2,4,5,6,8],[1,3,5,7,9],#Four-Five
            [1,2,3,4,5,6],[1,2,3,7,8,9],[4,5,6,7,8,9],[2,3,5,6,8,9],[1,3,4,6,7,9],[1,2,4,5,7,8],[2,3,4,6,7,8],[1,2,4,6,8,9],#Six-8
            [2,3,4,5,6,7,8,9],[1,3,4,5,6,7,8,9],[1,2,4,5,6,7,8,9],[1,2,3,5,6,7,8,9],[1,2,3,4,6,7,8,9],[1,2,3,4,5,7,8,9],[1,2,3,4,5,6,8,9],[1,2,3,4,5,6,7,9],[1,2,3,4,5,6,7,8]#Eight-9
            ]

for j,c_n_path in tqdm(enumerate(n_paras)):
    for i in range(20):
        c_octo_mask = n_mask(image_size,c_n_path)
        octo_n_masks.append(c_octo_mask)
octo_n_masks = np.array(octo_n_masks)
mega_masks[1340:,:,:] = octo_n_masks
#%% generate all mask infos.
np.savez_compressed('Masks_Metamer_Cut_v251011',masks=mega_masks)

