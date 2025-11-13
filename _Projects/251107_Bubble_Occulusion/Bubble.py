'''
Following the method of LeChang,2023,cell reports, I try to bubble my graph into different subparts, including the bubble and outside bubble


'''



#%%
import numpy as np
from PIL import Image, ImageDraw
import os
import random
import OS_Tools as ot
from tqdm import tqdm


# #Testrun part
# image_path = r'E:\#Coding_traces\Stim_Generator_bubble_single_LC\raw_fig\0003.jpg'
# image = Image.open(image_path).convert('RGB')
# image = image.resize((400, 400))
# width, height = image.size
# image_array = np.array(image)
# radius = 90
# num_masks = 80



#%% cutparts
def Bubble_Maker(image,radius=90,num_masks=80,width=400,height=400):
    image_array = np.array(image)
    all_bubbles = []
    centers = []
    for _ in range(num_masks):
        center_x = random.randint(radius, width - radius)
        center_y = random.randint(radius, height - radius)
        centers.append((center_x, center_y))

    for idx, (center_x, center_y) in enumerate(centers):
        # 创建圆形mask
        mask = Image.new('L', (width, height), 0)  # 'L'模式表示8位灰度图
        draw_mask = ImageDraw.Draw(mask)
        
        # 绘制圆形mask，内部为白色(255)，外部为黑色(0)
        draw_mask.ellipse([center_x - radius, center_y - radius, 
                        center_x + radius, center_y + radius], fill=255)
        # 将mask转换为numpy数组
        mask_array = np.array(mask)
        
        # 创建结果图像1：mask内的图片（mask外填灰色127）
        result_inside = image_array.copy()
        result_inside[mask_array == 0] = 127  # mask外部填灰色
        
        # 创建结果图像2：mask外的图片（mask内填灰色127）
        result_outside = image_array.copy()
        result_outside[mask_array == 255] = 127  # mask内部填灰色
        
        # 转换回PIL图像
        result_inside_img = Image.fromarray(result_inside.astype('uint8'))
        result_outside_img = Image.fromarray(result_outside.astype('uint8'))
        all_bubbles.append([image_array,(center_x, center_y),mask,result_inside_img,result_outside_img])
    # # 保存结果
    # result_inside_img.save(f'mask_results/inside_mask_{idx+1}.png')
    # result_outside_img.save(f'mask_results/outside_mask_{idx+1}.png')
    return all_bubbles


#%%
all_img_path = ot.Get_File_Name(r'E:\#Coding_traces\Stim_Generator_bubble_single_LC\raw_fig','.jpg')[:20]
savepath_occu = r'E:\#Coding_traces\Stim_Generator_bubble_single_LC\Occu'
savepath_rest = r'E:\#Coding_traces\Stim_Generator_bubble_single_LC\Rest'

generated_all_graph=[]

for i,c_img_path in tqdm(enumerate(all_img_path)):
    c_img = Image.open(c_img_path).convert('RGB')
    c_occ = Bubble_Maker(c_img,radius=90,num_masks=80)
    generated_all_graph.extend(c_occ)
ot.Save_Variable(r'E:\#Coding_traces\Stim_Generator_bubble_single_LC','Mask_Infos',generated_all_graph)
# a = Bubble_Maker(image_array,radius,num_masks)
# len(a)

#%% write bubbled graph into disk.

counter = 1
for i,c_img in tqdm(enumerate(generated_all_graph)):
    c_occu = c_img[-2]
    c_rest = c_img[-1]
    c_occu.save(ot.Join(savepath_occu,str(40000+counter))+'.jpg')
    c_rest.save(ot.Join(savepath_rest,str(50000+counter))+'.jpg')
    counter+=1