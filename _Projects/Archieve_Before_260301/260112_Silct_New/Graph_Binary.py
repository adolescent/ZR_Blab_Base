'''
AI 出来的图片需要洗一下，不然不干净
这个脚本用来对图片进行处理，二值化成灰色-黑色的。


'''
#%%
from PIL import Image
import numpy as np
import OS_Tools as ot
from tqdm import tqdm

def process_image_numpy(input_path, output_path, threshold=64, light_value=255):
    """
    使用NumPy高效处理图片
    
    参数:
    - input_path: 输入图片路径
    - output_path: 输出图片路径
    - threshold: 亮度阈值
    - light_value: 大于阈值的像素值
    """
    
    # 打开图片并转换为灰度
    img = Image.open(input_path)
    gray_img = img.convert('L')
    
    # 转换为NumPy数组
    gray_array = np.array(gray_img)
    
    # 根据阈值创建新数组
    # 小于阈值的设为0，大于等于阈值的设为127
    new_array = np.where(gray_array < threshold, 0, light_value).astype(np.uint8)

    # 将NumPy数组转换回PIL图像
    # new_img = Image.fromarray(new_array, mode='L').convert('RGB')
    new_img = Image.fromarray(new_array, mode='L')
    # 保存图片
    new_img.save(output_path,format='PNG',compress_level=0)
    # print(f"图片处理完成，已保存到: {output_path}")
    
    return new_img

#%% 使用示例
if __name__ == "__main__":
    
    input_image_path = ot.Get_File_Name(r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\Batch2_HD_Gray_Norm')
    output_folder = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\Batch2_HD_Gray_Norm_White'
    for i,c_graph in tqdm(enumerate(input_image_path)):
        c_filename = c_graph.split('\\')[-1]
        output_path = ot.Join(output_folder,c_filename)
        processed_image = process_image_numpy(c_graph, output_path)
