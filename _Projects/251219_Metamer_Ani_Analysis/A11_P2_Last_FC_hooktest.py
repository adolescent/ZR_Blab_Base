#%%
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import seaborn as sns
import numpy as np
import OS_Tools as ot
import os
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')

all_img_path = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Metamer_P4_C4321_Object_STI150_1300')
all_img_path.sort()
img_path = all_img_path[2]

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载预训练模型
model = models.alexnet(pretrained=True)
model.eval()  # 设置为评估模式

# 2. 准备图片
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)  # 添加batch维度

# 3. 定义钩子
activations = {}  # 存储激活值

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().clone()
    return hook

# 4. 注册钩子
# conv5是features模块的第10层（索引从0开始）
model.features[10].register_forward_hook(get_activation('conv5'))
# fc6是classifier模块的第0层
model.classifier[1].register_forward_hook(get_activation('fc6'))

# 5. 前向传播
with torch.no_grad():
    output = model(input_tensor)

# 6. 提取结果
conv5_features = activations['conv5']  # 形状: [1, 256, 13, 13]
fc6_features = activations['fc6']      # 形状: [1, 4096]

print(f"conv5 shape: {conv5_features.shape}")
print(f"fc6 shape: {fc6_features.shape}")
print(fc6_features.min())


#%% 使用示例
if __name__ == '__main__':
    # 假设 all_img_path 是包含1000张图片路径的列表

    # all_img_path = [...]  # 你的图片路径列表
    all_img_path = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Metamer_P4_C4321_Object_STI150_1300')
    all_img_path.sort()
    img_path = all_img_path[2]


    
#%%

# sns.heatmap(features['alexnet_fc7'].cpu().numpy()[:200,500:1000],center=0,vmax=5,vmin=-5,cmap='bwr')