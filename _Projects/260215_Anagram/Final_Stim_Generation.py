'''

Combine stim of anagram types.


'''


#%%
import OS_Tools as ot
from PIL import Image
from tqdm import tqdm
import numpy as np

savepath = r'Z:\Monkey_ephys\data_nas3\Zhangrui\Backups\Stimset_Generation\anagram_jigsaw_260224\260227_Anagram_Jigsaw'

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
batch_path1 = r'Z:\Monkey_ephys\data_nas3\Zhangrui\Backups\Stimset_Generation\anagram_jigsaw_260224\batch1'
batch_path2 = r'Z:\Monkey_ephys\data_nas3\Zhangrui\Backups\Stimset_Generation\anagram_jigsaw_260224\batch2'
batch_path3 = r'Z:\Monkey_ephys\data_nas3\Zhangrui\Backups\Stimset_Generation\anagram_jigsaw_260224\batch3'
batch_path4 = r'Z:\Monkey_ephys\data_nas3\Zhangrui\Backups\Stimset_Generation\anagram_jigsaw_260224\batch4'
batches = [batch_path1,batch_path2,batch_path3,batch_path4]

all_img = []
for i,c_batch in enumerate(batches):
    all_img += ot.Get_File_Name(c_batch,'.png')
all_img.sort()
#%% 
p1 = all_img[::2]
p2 = all_img[1::2]
# p1 三个循环
p1_n1 = p1[::3]
p1_n2 = p1[1::3]
p1_n3 = p1[2::3]
# p2 三个循环
p2_n1 = p2[::3]
p2_n2 = p2[1::3]
p2_n3 = p2[2::3]

p1_n1_s1 = p1_n1[::5]

#%% resize graph and rename.
counter = 1
for i in range(3):# 3 cycle
    c_p1 = [p1_n1,p1_n2,p1_n3][i]
    c_p2 = [p2_n1,p2_n2,p2_n3][i]
    for j in range(5):# 5 styles
        c_subtype_p1 = c_p1[j::5]
        c_subtype_p2 = c_p2[j::5]
        for k in tqdm(range(20)): # cycle objs.  
            #obj1
            c_img_a = c_subtype_p1[k]
            c_name_new_a = str(10000+counter)+'.jpg'
            save_filename_a = ot.Join(savepath,c_name_new_a)
            resize_image(c_img_a,save_filename_a)
            counter += 1
            #obj2
            c_img_b = c_subtype_p2[k]
            c_name_new_b = str(10000+counter)+'.jpg'
            save_filename_b = ot.Join(savepath,c_name_new_b)
            resize_image(c_img_b,save_filename_b)
            counter += 1


