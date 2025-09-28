'''
This script will cut graph into small pieces, and shuffle the range of each piece.

'''


#%%

import OS_Tools as ot
from tqdm import tqdm
import numpy as np
import copy
from PIL import Image
import random




wp = r'D:\#stimuli\Metamer_Mega_v250918\Base40'
all_path = ot.Get_File_Name(wp,'.jpg')
# input=Image.open(all_path[0])
savepath=r'D:\#stimuli\Metamer_Mega_v250918\Base40_Cut'
#%%


def shuffle_image_blocks(image, n_col, n_row):
    """
    将输入图像分块打乱重组，不调整图像尺寸，不能整除的部分直接裁剪忽略
    
    Args:
        image: PIL Image对象或图像路径
        n_col: 水平方向分块数
        n_row: 垂直方向分块数
        
    Returns:
        PIL Image对象: 打乱重组后的图像
    """
    # 如果输入是路径，打开图像
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image.copy()
    
    # 转换为RGB（确保处理彩色和灰度图像一致）
    img = img.convert('RGB')
    img_array = np.array(img)
    
    # 获取图像尺寸
    height, width = img_array.shape[0], img_array.shape[1]
    
    # 计算每个块的尺寸（向下取整）
    block_height = height // n_row
    block_width = width // n_col
    
    # 计算裁剪后的尺寸
    cropped_height = block_height * n_row
    cropped_width = block_width * n_col
    
    # 裁剪图像（忽略不能整除的部分）
    img_array = img_array[:cropped_height, :cropped_width]
    
    # 提取所有图像块
    blocks = []
    for i in range(n_row):
        for j in range(n_col):
            y_start = i * block_height
            y_end = y_start + block_height
            x_start = j * block_width
            x_end = x_start + block_width
            block = img_array[y_start:y_end, x_start:x_end]
            blocks.append(block)
    
    # 随机打乱块顺序
    random.shuffle(blocks)
    
    # 创建新图像
    new_img_array = np.zeros_like(img_array)
    index = 0
    for i in range(n_row):
        for j in range(n_col):
            y_start = i * block_height
            y_end = y_start + block_height
            x_start = j * block_width
            x_end = x_start + block_width
            new_img_array[y_start:y_end, x_start:x_end] = blocks[index]
            index += 1
    
    return Image.fromarray(new_img_array)

def nested_shuffle_fixed_outer(image, outer_n_col=2, outer_n_row=2, inner_n_col=3, inner_n_row=3):
    """
    将图像分割成外层块，然后对每个外层块内部进行分块打乱，外层块位置保持不变
    
    Args:
        image: PIL Image对象或图像路径
        outer_n_col: 外层水平方向分块数
        outer_n_row: 外层垂直方向分块数
        inner_n_col: 内层水平方向分块数
        inner_n_row: 内层垂直方向分块数
        
    Returns:
        PIL Image对象: 嵌套打乱重组后的图像
    """
    # 如果输入是路径，打开图像
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image.copy()
    
    # 转换为RGB（确保处理彩色和灰度图像一致）
    img = img.convert('RGB')
    img_array = np.array(img)
    
    # 获取图像尺寸
    height, width = img_array.shape[0], img_array.shape[1]
    
    # 计算外层每个块的尺寸（向下取整）
    outer_block_height = height // outer_n_row
    outer_block_width = width // outer_n_col
    
    # 计算裁剪后的尺寸
    cropped_height = outer_block_height * outer_n_row
    cropped_width = outer_block_width * outer_n_col
    
    # 裁剪图像（忽略不能整除的部分）
    img_array = img_array[:cropped_height, :cropped_width]
    
    # 创建新图像
    new_img_array = np.zeros_like(img_array)
    
    # 处理每个外层块
    for i in range(outer_n_row):
        for j in range(outer_n_col):
            # 提取外层块
            y_start = i * outer_block_height
            y_end = y_start + outer_block_height
            x_start = j * outer_block_width
            x_end = x_start + outer_block_width
            outer_block = img_array[y_start:y_end, x_start:x_end]
            
            # 将外层块转换为PIL图像
            outer_block_img = Image.fromarray(outer_block)
            
            # 对外层块内部进行分块打乱
            shuffled_inner_block = shuffle_image_blocks(
                outer_block_img, inner_n_col, inner_n_row
            )
            
            # 将打乱后的内层块转换为数组
            shuffled_array = np.array(shuffled_inner_block)
            
            # 确保打乱后的块尺寸与外层块尺寸匹配
            # 如果尺寸不匹配，调整打乱后的块尺寸
            if shuffled_array.shape != outer_block.shape:
                # 取最小尺寸
                min_height = min(shuffled_array.shape[0], outer_block_height)
                min_width = min(shuffled_array.shape[1], outer_block_width)
                
                # 裁剪打乱后的块以匹配外层块尺寸
                shuffled_array = shuffled_array[:min_height, :min_width]
                
                # 如果打乱后的块太小，用黑色填充
                if shuffled_array.shape[0] < outer_block_height or shuffled_array.shape[1] < outer_block_width:
                    padded_array = np.zeros_like(outer_block)
                    padded_array[:shuffled_array.shape[0], :shuffled_array.shape[1]] = shuffled_array
                    shuffled_array = padded_array
            
            # 将打乱后的内层块放回新图像
            new_img_array[y_start:y_end, x_start:x_end] = shuffled_array
    
    return Image.fromarray(new_img_array)


#%%
# a=shuffle_image_blocks(input, n_col=16, n_row=16)


def remove_zero_grid_and_pad(image):
    """
    移除图像中的所有全0行和列，并在剩余区域填充127保持原始大小
    
    参数:
        image: PIL Image对象 (RGB格式)
    
    返回:
        处理后的PIL Image对象
    """
    # 将图像转换为NumPy数组
    img_array = np.array(image)
    detector = img_array.mean(-1)
    height,width = detector.shape
    result_image = np.zeros(shape = img_array.shape,dtype='u1')
    # cycle all line
    fill_h=0
    counter=0
    for i in range(height):
        c_line = detector[i,:]
        if c_line.sum()!=0:
            result_image[counter,:,:] = img_array[i,:,:]
            counter+=1
        else:
            fill_h+=1

    img_array = result_image
    counter=0
    fill_v=0
    for i in range(width):
        c_line = detector[:,i]
        if c_line.sum()!=0:
            result_image[:,counter,:] = img_array[:,i,:]
            counter+=1
        else:
            fill_v +=1
    if fill_h!=0:
        result_image[-fill_h:,:,:]=[127,127,127]
    if fill_v !=0:
        result_image[:,-fill_v:,:]=[127,127,127]
    result_image = Image.fromarray(result_image)
    
    return result_image
#%%

output_image = nested_shuffle_fixed_outer(
        input, 
        outer_n_col=2, 
        outer_n_row=2, 
        inner_n_col=2, 
        inner_n_row=2
    )
output_image = remove_zero_grid_and_pad(output_image)
output_image

#%% get cut-shuffled graph

counter=1
# C4S3
for N in range(2):
    for i,c_path in enumerate(all_path):
        
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=4, 
            outer_n_row=4, 
            inner_n_col=3, 
            inner_n_row=3
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1

    # C3S4
    for i,c_path in enumerate(all_path):
        
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=3, 
            outer_n_row=3, 
            inner_n_col=4, 
            inner_n_row=4
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1

    # C2S6
    for i,c_path in enumerate(all_path):
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=2, 
            outer_n_row=2, 
            inner_n_col=6, 
            inner_n_row=6
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1
    # C1S12
    for i,c_path in enumerate(all_path):
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=1, 
            outer_n_row=1, 
            inner_n_col=12, 
            inner_n_row=12
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1
    ## another batch,8-9cuts
    # C4S2
    for i,c_path in enumerate(all_path):
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=4, 
            outer_n_row=4, 
            inner_n_col=2, 
            inner_n_row=2
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1
    # C3S3
    for i,c_path in enumerate(all_path):
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=3, 
            outer_n_row=3, 
            inner_n_col=3, 
            inner_n_row=3
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1
    # C2S4
    for i,c_path in enumerate(all_path):
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=2, 
            outer_n_row=2, 
            inner_n_col=4, 
            inner_n_row=4
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1
    # C1S8
    for i,c_path in enumerate(all_path):
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=1, 
            outer_n_row=1, 
            inner_n_col=8, 
            inner_n_row=8
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1

    ## third batch, bigger cuts
    # C1S4
    for i,c_path in enumerate(all_path):
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=1, 
            outer_n_row=1, 
            inner_n_col=4, 
            inner_n_row=4
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1
    # C2S2
    for i,c_path in enumerate(all_path):
        c_input = Image.open(c_path)
        # save C4 big 
        output_image=nested_shuffle_fixed_outer(
            c_input, 
            outer_n_col=2, 
            outer_n_row=2, 
            inner_n_col=2, 
            inner_n_row=2
        )
        c_name = str(50000+counter)+'.jpg'
        output_image = remove_zero_grid_and_pad(output_image)
        output_image.save(ot.Join(savepath,c_name))
        counter+=1