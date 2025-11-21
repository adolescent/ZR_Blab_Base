'''
This will cut graph into 8 parts from the center, then 
'''




#%%

import OS_Tools as ot
from Spike_Tools import *
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


wp = r'D:\#stimuli\Metamer_Cut_v251011\Base40'
original_image = Image.open(ot.Join(wp,'0012.jpg')).convert('RGB')

#%% function of generate mask of 


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

def apply_mask(img,mask):
    '''
    Remember that mask must be in the same shape
    
    '''

    masked_img = Image.new('RGB', img.size, (127,127,127))
    masked_img.paste(img, (0, 0), mask)

    return masked_img

#%% add octo mask to given graphs.
all_imgs = ot.Get_File_Name(wp,'.jpg')[:20]

octo_path = r'D:\#stimuli\Metamer_Cut_v251011\Octo_cut'
n_path = r'D:\#stimuli\Metamer_Cut_v251011\N_cut'
image_size=(400,400)

# apply different cut methods to imgs, save them into different folder.
octo_paras = [[1],[2],[3],[4],[5],[6],[7],[8],#8
              [1,2],[3,4],[5,6],[7,8],[1,5],[2,6],[3,7],[4,8],#8
              [1,3,5,7],[2,4,6,8],[1,2,3,4],[5,6,7,8],[3,4,5,6],[1,2,7,8],#6
              [3,4,5,6,7,8],[1,2,5,6,7,8],[1,2,3,4,7,8],[1,2,3,4,5,6],#4
              [2,3,4,5,6,7,8],[1,3,4,5,6,7,8],[1,2,4,5,6,7,8],[1,2,3,5,6,7,8],[1,2,3,4,6,7,8],[1,2,3,4,5,7,8],[1,2,3,4,5,6,8],[1,2,3,4,5,6,7]#8
              ]


counter=1
for j,c_octo_para in tqdm(enumerate(octo_paras)):
    for i,c_img_name in enumerate(all_imgs):
        
        c_img = Image.open(c_img_name)
        c_octo_mask = octo_mask(image_size,c_octo_para)
        c_masked_img = apply_mask(c_img,c_octo_mask)
        c_name = str(10000+counter)[1:]+'.jpg'
        c_masked_img.save(ot.Join(octo_path,c_name))
        counter+=1

#%% add n mask to given graphs
all_imgs = ot.Get_File_Name(wp,'.jpg')[:20]

n_path = r'D:\#stimuli\Metamer_Cut_v251011\N_cut'
image_size=(400,400)

n_paras = [[1],[2],[3],[4],[5],[6],[7],[8],[9],#One-9
            [1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9],[1,5,9],[3,5,7],#Three-8
            [1,3,7,9],[2,4,6,8],[2,4,5,6,8],[1,3,5,7,9],#Four-Five
            [1,2,3,4,5,6],[1,2,3,7,8,9],[4,5,6,7,8,9],[2,3,5,6,8,9],[1,3,4,6,7,9],[1,2,4,5,7,8],[2,3,4,6,7,8],[1,2,4,6,8,9],#Six-8
            [2,3,4,5,6,7,8,9],[1,3,4,5,6,7,8,9],[1,2,4,5,6,7,8,9],[1,2,3,5,6,7,8,9],[1,2,3,4,6,7,8,9],[1,2,3,4,5,7,8,9],[1,2,3,4,5,6,8,9],[1,2,3,4,5,6,7,9],[1,2,3,4,5,6,7,8]#Eight-9
            ]


counter=1
for j,c_n_path in tqdm(enumerate(n_paras)):
    for i,c_img_name in enumerate(all_imgs):

        c_img = Image.open(c_img_name)
        c_mask = n_mask(image_size,c_n_path)
        c_masked_img = apply_mask(c_img,c_mask)
        c_name = str(10000+counter)[1:]+'.jpg'
        c_masked_img.save(ot.Join(n_path,c_name))
        counter+=1

        