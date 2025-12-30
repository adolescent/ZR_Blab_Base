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
# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 加载预训练模型
        self.alexnet = models.alexnet(pretrained=True).to(device).eval()
        self.vgg19 = models.vgg19(pretrained=True).to(device).eval()
        self.resnet18 = models.resnet18(pretrained=True).to(device).eval()
    
    def extract_features(self, img_path):
        """提取单张图片的特征"""
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(device)
        
        features = {}
        
        with torch.no_grad():
            # AlexNet特征提取
            x = self.alexnet.features(img_tensor)
            x = self.alexnet.avgpool(x)
            x = torch.flatten(x, 1)
            
            # 手动获取fc7层的输出（ReLU之前）
            x = self.alexnet.classifier[0](x)  # Dropout (训练时使用，评估时忽略)
            x = self.alexnet.classifier[1](x)  # Linear 1
            x = self.alexnet.classifier[2](x)  # ReLU 1
            x = self.alexnet.classifier[3](x)  # Dropout
            x = self.alexnet.classifier[4](x)  # Linear 2 (fc7) - 保存这个
            # x = self.alexnet.classifier[5](x)
            features['alexnet_fc7'] = x.clone().cpu()
            
            # VGG19特征提取
            x_vgg = self.vgg19.features(img_tensor)
            x_vgg = self.vgg19.avgpool(x_vgg)
            x_vgg = torch.flatten(x_vgg, 1)
            
            # 手动获取fc2层的输出（ReLU之前）
            x_vgg = self.vgg19.classifier[0](x_vgg)  # Linear 1
            x_vgg = self.vgg19.classifier[1](x_vgg)  # ReLU 1
            x_vgg = self.vgg19.classifier[2](x_vgg)  # Dropout
            x_vgg = self.vgg19.classifier[3](x_vgg)  # Linear 2 - 保存这个
            # x_vgg = self.vgg19.classifier[4](x_vgg)  # ReLU 2
            # x_vgg = self.vgg19.classifier[5](x_vgg)  # Dropout
            # x_vgg = self.vgg19.classifier[6](x_vgg)  # Linear 3 (fc2) 
            features['vgg19_fc2'] = x_vgg.clone().cpu()
            
            # ResNet18特征提取
            x_res = self.resnet18.conv1(img_tensor)
            x_res = self.resnet18.bn1(x_res)
            x_res = self.resnet18.relu(x_res)
            x_res = self.resnet18.maxpool(x_res)
            x_res = self.resnet18.layer1(x_res)
            x_res = self.resnet18.layer2(x_res)
            x_res = self.resnet18.layer3(x_res)
            x_res = self.resnet18.layer4(x_res)
            x_res = self.resnet18.avgpool(x_res)
            features['resnet18_pool'] = torch.flatten(x_res, 1).cpu()
        
        return features

# 批量提取特征
def extract_all_features(all_img_path, batch_size=128):
    extractor = FeatureExtractor()
    all_features = {
        'alexnet_fc7': [],
        'vgg19_fc2': [],
        'resnet18_pool': []
    }
    
    for i in range(0, len(all_img_path), batch_size):
        batch_paths = all_img_path[i:i+batch_size]
        
        for img_path in batch_paths:
            features = extractor.extract_features(img_path)
            for key in features:
                all_features[key].append(features[key])
        
        print(f'Processed {min(i+batch_size, len(all_img_path))}/{len(all_img_path)} images')
    
    # 转换为张量
    for key in all_features:
        all_features[key] = torch.cat(all_features[key], dim=0)
    
    return all_features

#%% 使用示例
if __name__ == '__main__':
    # 假设 all_img_path 是包含1000张图片路径的列表

    # all_img_path = [...]  # 你的图片路径列表
    all_img_path = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Metamer_P4_C4321_Object_STI150_1300')
    all_img_path.sort()
    
    # 提取特征
    features = extract_all_features(all_img_path)
    
    # 打印特征维度
    for model_name, feature in features.items():
        print(f"{model_name}: {feature.shape}")
    print(features['alexnet_fc7'].min())
    print(features['vgg19_fc2'].min())
    print(features['resnet18_pool'].min())
    
#%%
    np.save(f"alexnet_fc7_features.npy", features['alexnet_fc7'].numpy())
    np.save(f"vgg19_fc2_features.npy", features['vgg19_fc2'].numpy())
    np.save(f"res18_pool_features.npy", features['resnet18_pool'].numpy())
    