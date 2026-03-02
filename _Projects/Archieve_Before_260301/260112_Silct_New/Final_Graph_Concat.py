'''
最终用于生成刺激集的

1-400 HD化的silct
401-800 Boulder
801-1200 平衡对比度后的silct
1201-2400 三组写实化的像素对应
2401-3200 两组语义相同的新刺激
3200-3500 两遍sti150 localizer
'''

#%%
import OS_Tools as ot
from PIL import Image
from tqdm import tqdm

savepath = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\Doodle_AI_v260121'
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



#%%     1 Series as Doodle
doodle_path = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\_Doodle_HD_Gray_Norm'
c_files = ot.Get_File_Name(doodle_path)
for i,c_file in tqdm(enumerate(c_files)):
    c_name_new = str(10001+i)+'.jpg'
    save_filename = ot.Join(savepath,c_name_new)
    resize_image(c_file,save_filename)
#%%     2 Series as Boulder
boulder_path = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\_Doodle_HD_Boulder'
c_files = ot.Get_File_Name(boulder_path)
for i,c_file in tqdm(enumerate(c_files)):
    c_name_new = str(20001+i)+'.jpg'
    save_filename = ot.Join(savepath,c_name_new)
    resize_image(c_file,save_filename)
#%%     3 Series as Silct
silct_path = r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\_Doodle_HD_Silct_Norm'
c_files = ot.Get_File_Name(silct_path)
for i,c_file in tqdm(enumerate(c_files)):
    c_name_new = str(30001+i)+'.jpg'
    save_filename = ot.Join(savepath,c_name_new)
    resize_image(c_file,save_filename)
#%%     4 Series as AI Pix aligned regeneration.
ai_1 = ot.Get_File_Name(r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\_AI_Real_1_Size_adj_GrayBK')
ai_2 = ot.Get_File_Name(r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\_AI_Real_2_Size_adj_GrayBK')
ai_3 = ot.Get_File_Name(r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\_AI_Real_3_Size_adj_GrayBK')
ai_pix_alinged = ai_1+ai_2+ai_3
for i,c_file in tqdm(enumerate(ai_pix_alinged)):
    c_name_new = str(40001+i)+'.jpg'
    save_filename = ot.Join(savepath,c_name_new)
    resize_image(c_file,save_filename)

#%%     5 Series as AI Syntax Regeneration
ai_syn1 = ot.Get_File_Name(r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Regeneration_1_GrayBK')
ai_syn2 = ot.Get_File_Name(r'E:\#Stimsets\Silct\Silct_New_260112\Raw_Doodle\AI_Regeneration_2_GrayBK')

ai_syn_alinged = ai_syn1+ai_syn2
for i,c_file in tqdm(enumerate(ai_syn_alinged)):
    c_name_new = str(50001+i)+'.jpg'
    save_filename = ot.Join(savepath,c_name_new)
    resize_image(c_file,save_filename)
#%%     0 series as FOB. use STI150*2

