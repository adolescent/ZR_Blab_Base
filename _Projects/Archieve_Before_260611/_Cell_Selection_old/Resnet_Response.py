'''
得到resnet的层数对全部图片的响应。

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

#%% 3. 加载模型并提取中间层
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
resnet50.eval()

# 我们只需要到 layer4 和 avgpool，不需要最后的 fc 层
# 使用子模块提取
conv_layer = nn.Sequential(*list(resnet50.children())[:-2]) # 截取到 layer4
pool_layer = nn.Sequential(*list(resnet50.children())[:-1]) # 截取到 avgpool


#%% 4. 开始提取
all_avgpool = []
all_lastconv = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        
        # 提取 Layer4 (Last Conv) -> [Batch, 2048, 7, 7]
        conv_out = conv_layer(batch)
        all_lastconv.append(conv_out.cpu().numpy())
        # 提取 AvgPool -> [Batch, 2048, 1, 1]
        pool_out = pool_layer(batch)
        # 展平为 [Batch, 2048]
        pool_out = pool_out.view(pool_out.size(0), -1)
        all_avgpool.append(pool_out.cpu().numpy())


#%% 5. 合并为最终矩阵
matrix_avgpool = np.concatenate(all_avgpool, axis=0)      # (1000, 2048)
matrix_lastconv = np.concatenate(all_lastconv, axis=0)    # (1000, 2048, 7, 7)

print(f"AvgPool Matrix Shape: {matrix_avgpool.shape}")
print(f"Last Conv Matrix Shape: {matrix_lastconv.shape}")


#%% save dcnn response on given layer.
savepath = r'E:\#Preprocessed_Data\Selected_Cells'

np.savez_compressed(
    ot.Join(savepath,'Res50_Response'), 
    avgpool=matrix_avgpool, 
    last_conv=matrix_lastconv
)
