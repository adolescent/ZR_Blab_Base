
'''
和resnet版本一样，不过用的是Alexnet
得到了last conv，fc6和fc7的结果，

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
savepath = r'E:\#Preprocessed_Data\Selected_Cells'
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
alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1').to(device)
alexnet.eval()

#%% 4. 指定提取节点
# 提示：AlexNet 的 features 包含 13 个子层 (0-12)
# classifier 包含 7 个子层 (0-6)
return_nodes = {
    'features.12': 'last_conv',  # 最后一个卷积层响应 [256, 6, 6]
    'classifier.2': 'fc6',       # 第一个 FC 激活后 [4096]
    'classifier.5': 'fc7'        # 第二个 FC 激活后 [4096]
}

fetcher = create_feature_extractor(alexnet, return_nodes=return_nodes)

#  容器
all_conv = []
all_fc6 = []
all_fc7 = []

#%% 5. 从你的 dataloader 中提取
print("开始提取 AlexNet 特征...")
with torch.no_grad():
    for batch_data in loader:
        images = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
        images = images.to(device)
        
        # 执行提取
        features = fetcher(images)
        
        # 存入列表
        all_conv.append(features['last_conv'].cpu().numpy())
        all_fc6.append(features['fc6'].cpu().numpy())
        all_fc7.append(features['fc7'].cpu().numpy())

#  合并
matrix_conv = np.concatenate(all_conv, axis=0) # (1000, 256, 6, 6)
matrix_fc6  = np.concatenate(all_fc6, axis=0)  # (1000, 4096)
matrix_fc7  = np.concatenate(all_fc7, axis=0)  # (1000, 4096)

#%% 6. 保存为 npz
save_filename = ot.Join(savepath,"Alex_Response.npz")
np.savez_compressed(
    save_filename, 
    fc6 = matrix_fc6,
    fc7 = matrix_fc7, 
    last_conv=matrix_conv
)

print(f"提取完成！文件已保存至: {save_filename}")
print(f"Conv 形状: {matrix_conv.shape}")
print(f"FC6 形状: {matrix_fc6.shape}")