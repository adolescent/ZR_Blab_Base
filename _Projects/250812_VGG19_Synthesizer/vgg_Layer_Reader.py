'''
Load in graph and getting it's vgg layer representation.

'''

#%%
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import os 


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# input_image = Image.open(filename)
input_image = Image.open('44.jpg')
spatial_constraint=(1, 1)
feature_layers=['pool1', 'pool2', 'pool4']
device='cuda'
learning_rate=0.1
num_steps=500
vgg = models.vgg19(weights=True).features.to(device).eval()
vgg.requires_grad_(False)
#%%
preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
target_tensor = preprocess(input_image).unsqueeze(0).to(device)
target_size = (224, 224)

layer_index_map = {
            'pool1': 4,   # 第一个池化层
            'pool2': 9,   # 第二个池化层
            'pool3': 18,  # 第三个池化层
            'pool4': 27,  # 第四个池化层
            'pool5': 36   # 第五个池化层
        }

#%% 定义钩子，钩取特定层的输出
## register hook

outputs = {}

# 4. 定义钩子函数
def hook_fn(module, input, output, key):
    outputs[key] = output

# 5. 为所有卷积层和全连接层注册钩子
hooks = {}
for name, layer in vgg.named_modules():
    # 选择你感兴趣的层（这里以卷积层和全连接层为例）
    # if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
    hooks[name] = layer.register_forward_hook(
        lambda m, i, o, n=name: hook_fn(m, i, o, n)
    )


output = vgg(target_tensor)





#%%
sns.heatmap(outputs['4'][0,-2,:,:].cpu().numpy(),center=0)
