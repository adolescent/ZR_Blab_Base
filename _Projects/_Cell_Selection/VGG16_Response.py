'''
和resnet版本一样，不过用的是VGG16
得到了last conv，fc1和fc2的结果，
分别对应last conv，fc6和fc7
'''


#%%
import seaborn as sns
import OS_Tools as ot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from torchvision.models.feature_extraction import create_feature_extractor

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

all_filename = ot.Get_File_Name(r'E:\#Stimsets\Metamer_P4_C4321_Object_STI150_1300')
all_filename.sort()
all_metamer_filepath = all_filename[:1000]

#%% 1. 定义简单的 Dataset 类
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # 确保输出是 7x7
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

#%% 2. 准备数据
image_paths = all_metamer_filepath # 替换为你的 1000 张路径
dataset = ImageDataset(image_paths)
loader = DataLoader(dataset, batch_size=32, shuffle=False)


#%% 3. 准备模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1').to(device)
vgg16.eval()

#%% 4. 指定提取节点
# 我们可以通过 print(dict(vgg16.named_modules()).keys()) 查看所有节点名称
return_nodes = {
    'features.30': 'last_conv',  # 卷积层响应 [512, 7, 7]
    'classifier.0': 'fc1',       # 全连接1 [4096]
    'classifier.3': 'fc2'        # 全连接2 [4096]
}

fetcher = create_feature_extractor(vgg16, return_nodes=return_nodes)

# 容器：用于存放最终结果
all_conv = []
all_fc1 = []
all_fc2 = []

#%%
# 5. 从你现有的 dataloader 中提取
print("开始提取特征...")
with torch.no_grad():
    for batch_idx, batch_data in enumerate(loader):
        # 兼容处理：有些 dataloader 返回 (img, label)，有些只返回 img
        images = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
        images = images.to(device)
        
        # 得到特征字典
        features = fetcher(images)
        
        # 存入列表 (转回 CPU 避免显存溢出)
        all_conv.append(features['last_conv'].cpu().numpy())
        all_fc1.append(features['fc1'].cpu().numpy())
        all_fc2.append(features['fc2'].cpu().numpy())
        
        if (batch_idx + 1) % 5 == 0:
            print(f"已处理 {batch_idx + 1} 个 Batch")

#%% 6. 合并为大矩阵
matrix_conv = np.concatenate(all_conv, axis=0) # (1000, 512, 7, 7)
matrix_fc1  = np.concatenate(all_fc1, axis=0)  # (1000, 4096)
matrix_fc2  = np.concatenate(all_fc2, axis=0)  # (1000, 4096)

#%% 7. 保存为 npz
savepath = r'E:\#Preprocessed_Data\Selected_Cells'
save_filename = ot.Join(savepath,"VGG16_Response.npz")
np.savez_compressed(
    save_filename, 
    fc1 = matrix_fc1,
    fc2 = matrix_fc2, 
    last_conv=matrix_conv
)

print(f"提取完成！文件已保存至: {save_filename}")
print(f"Conv 形状: {matrix_conv.shape}")
print(f"FC1 形状: {matrix_fc1.shape}")


