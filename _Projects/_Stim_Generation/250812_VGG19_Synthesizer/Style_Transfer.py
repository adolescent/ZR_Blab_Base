'''

It seems that we might need a style transfer method for graph generation...

'''

#%%

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os 
import OS_Tools as ot

# 设备配置

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像加载和预处理
def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    return image.to(device)

# 将Tensor转换为图像
def tensor_to_image(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
    return image

# 显示图像
def imshow(tensor, title=None):
    image = tensor_to_image(tensor)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

# 计算Gram矩阵（风格特征）
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (d * h * w)

# 加载预训练VGG19并修改网络
def get_vgg_model():
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    
    # 替换MaxPool为AvgPool
    for i, layer in enumerate(vgg):
        if isinstance(layer, nn.MaxPool2d):
            vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    
    # 冻结所有参数
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    return vgg

# 获取目标层的输出
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'4': 'pool1', '9': 'pool2', '27': 'pool4', '21': 'conv4_2'}
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    
    return features

# 主函数
def style_transfer(content_path, style_path, num_iterations=500, 
                   content_weight=1, style_weight=1e6,
                   style_layer_weights={'pool1': 1, 'pool2': 2, 'pool4': 5}):
    # 加载图像
    content = load_image(content_path)
    style = load_image(style_path, shape=content.shape[-2:])
    
    # 初始化目标图像（使用内容图像作为起点）
    target = content.clone().requires_grad_(True)
    
    # 获取VGG模型
    vgg = get_vgg_model()
    
    # 获取特征
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    
    # 计算风格特征的Gram矩阵
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features 
                   if layer in style_layer_weights}
    
    # 优化器 (Adam优化目标图像)
    optimizer = optim.Adam([target], lr=0.05)
    
    # 存储最佳结果
    best_loss = float('inf')
    best_target = None
    
    print('开始风格迁移...')
    for i in range(1, num_iterations+1):
        # 前向传播获取目标图像特征
        target_features = get_features(target, vgg)
        
        # 初始化损失
        content_loss = 0
        style_loss = 0
        
        # 计算内容损失 (conv4_2层)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        # 计算风格损失 (pool1, pool2, pool4层)
        for layer in style_layer_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            
            layer_loss = style_layer_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_loss
        
        # 总损失
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 更新最佳结果
        if total_loss < best_loss:
            best_loss = total_loss
            best_target = target.clone()
        
        # 每200次迭代显示进度
        if i % 200 == 0:
            print(f'迭代 [{i}/{num_iterations}] 总损失: {total_loss.item():.4f}, '
                  f'内容损失: {content_loss.item():.4f}, 风格损失: {(style_weight * style_loss).item():.4f}')
    
    # 返回最佳结果
    return best_target
#%%
# 设置路径和参数
wp = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\real_stim'

# content_path = ot.Join(wp,"0241.jpg")  # 替换为你的内容图像路径
# style_path = ot.Join(wp,"0001.jpg")     # 替换为你的风格图像路径

content_path = 'synth.jpg'
style_path = '0015.jpg'

# 执行风格迁移
result = style_transfer(
    content_path=content_path,
    style_path=style_path,
    num_iterations=1000,
    content_weight=1,
    style_weight=1e4,
    style_layer_weights={'pool1': 1, 'pool2': 2, 'pool4': 5}
)

# 显示和保存结果
imshow(result, "Transfer style")
result_image = tensor_to_image(result)
# result_image
# plt.imsave("transfer_11_12.jpg", result_image)