'''
提取特定中间层的反应,文中使用的是最后一个卷积层,那么就如其所愿。

稍微修改可以提取最后一个全连接层
'''





#%%
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
import numpy as np
import OS_Tools as ot
import seaborn as sns
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')


class BatchFeatureExtractor:
    def __init__(self, model_name='vgg19', batch_size=32, device=None):
        """
        初始化批量特征提取器
        
        Args:
            model_name: 模型名称，可选 'alexnet', 'vgg19', 'resnet18'
            batch_size: 批量大小
            device: 指定设备，默认为自动检测
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 加载模型并注册钩子
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 存储特征
        self.features = {}
        
        # 注册钩子
        self._register_hook()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 存储特征矩阵
        self.feature_matrix = None
    
    def _load_model(self):
        """加载预训练模型"""
        if self.model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif self.model_name == 'vgg19':
            model = models.vgg19(pretrained=True)
        elif self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        else:
            raise ValueError(f"不支持模型: {self.model_name}")
        return model
    
    def _get_activation_hook(self, layer_name):
        """创建钩子函数来捕获特定层的输出"""
        def hook(model, input, output):
            self.features[layer_name] = output.detach().clone()
        return hook
    
    def _register_hook(self):
        """注册钩子到最后一个卷积层"""
        if self.model_name == 'alexnet':
            # AlexNet: 提取最后一个卷积层后的ReLU输出
            target_layer = self.model.features[11]  # ReLU after conv5
        elif self.model_name == 'vgg19':
            # VGG19: 提取最后一个卷积层后的ReLU输出
            target_layer = self.model.features[36]  # ReLU after conv5_4
        elif self.model_name == 'resnet18':
            # ResNet18: 提取layer4中最后一个卷积层后的ReLU输出
            target_layer = self.model.layer4[1]
        
        # 注册前向钩子
        target_layer.register_forward_hook(self._get_activation_hook('last_conv'))
    
    def extract_batch_features(self, img_paths):
        """
        批量提取图像特征
        
        Args:
            img_paths: 图像路径列表
            
        Returns:
            feature_matrix: 特征矩阵 (n_images x n_features)
        """
        n_images = len(img_paths)
        
        # 创建数据加载器
        dataset = ImageDataset(img_paths, self.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # 存储所有特征
        all_features = []
        
        # 批量处理图像
        with torch.no_grad():
            for batch_images in tqdm(dataloader, desc=f"提取{self.model_name}特征"):
                batch_images = batch_images.to(self.device)
                
                # 前向传播
                _ = self.model(batch_images)
                
                # 获取特征
                if 'last_conv' in self.features:
                    batch_features = self.features['last_conv']
                    
                    # 展平特征 (batch_size, C, H, W) -> (batch_size, C*H*W)
                    batch_features_flat = batch_features.view(batch_features.size(0), -1)
                    
                    # 转移到CPU并添加到列表
                    all_features.append(batch_features_flat.cpu())
                    
                    # 清空当前特征（可选）
                    self.features.clear()
                else:
                    raise RuntimeError("未能提取到特征")
        
        # 合并所有批次的特征
        if all_features:
            feature_matrix = torch.cat(all_features, dim=0)
            self.feature_matrix = feature_matrix.numpy()
            return self.feature_matrix
        else:
            raise RuntimeError("没有提取到任何特征")
    
    def get_feature_matrix(self):
        """获取特征矩阵"""
        if self.feature_matrix is not None:
            return self.feature_matrix
        else:
            raise RuntimeError("请先调用extract_batch_features方法提取特征")


class ImageDataset(torch.utils.data.Dataset):
    """自定义图像数据集"""
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"加载图像 {img_path} 时出错: {e}")
            # 返回一个空图像作为占位符
            return torch.zeros(3, 224, 224) if self.transform else None


def extract_all_models_features(all_img_paths, batch_size=32):
    """
    从所有三个模型中提取特征
    
    Args:
        all_img_paths: 图像路径列表
        batch_size: 批量大小
        
    Returns:
        dict: 包含三个模型特征矩阵的字典
    """
    models_to_extract = ['alexnet', 'vgg19', 'resnet18']
    feature_matrices = {}
    
    for model_name in models_to_extract:
        print(f"\n{'='*50}")
        print(f"正在提取 {model_name.upper()} 特征...")
        print(f"{'='*50}")
        
        # 创建特征提取器
        extractor = BatchFeatureExtractor(
            model_name=model_name, 
            batch_size=batch_size
        )
        
        # 提取特征
        feature_matrix = extractor.extract_batch_features(all_img_paths)
        
        # 存储特征矩阵
        feature_matrices[model_name] = feature_matrix
        
        # 打印特征矩阵信息
        print(f"{model_name.upper()} 特征矩阵形状: {feature_matrix.shape}")
        print(f"{model_name.upper()} 特征维度: {feature_matrix.shape[1]}")
        
        # 可选：保存特征矩阵到文件
        np.save(f"{model_name}_features.npy", feature_matrix)
        print(f"{model_name.upper()} 特征已保存到 {model_name}_features.npy")
        
        # 释放GPU内存
        torch.cuda.empty_cache()
    
    return feature_matrices


def main(all_img_paths):
    # 示例：假设你有1000张图片的路径列表
    # all_img_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
    
    # 为了演示，创建一个模拟的图像路径列表
    # 在实际使用中，请替换为你的真实图像路径
    # all_img_paths = ["your_image_path_{}.jpg".format(i) for i in range(1000)]
    
    # 或者从文本文件中读取
    # with open('image_paths.txt', 'r') as f:
    #     all_img_paths = [line.strip() for line in f.readlines()]
    
    # 这里使用一个示例路径（你需要替换为实际的路径列表）
    # all_img_paths = []
    if len(all_img_paths) == 0:
        print("请提供图像路径列表")
        return
    
    print(f"总共 {len(all_img_paths)} 张图片")
    
    # 设置批量大小（根据GPU内存调整）
    batch_size = 32
    
    # 提取所有模型的特征
    feature_matrices = extract_all_models_features(all_img_paths, batch_size)
    
    # 打印总结信息
    print("\n" + "="*60)
    print("特征提取完成！")
    print("="*60)
    
    for model_name, matrix in feature_matrices.items():
        print(f"{model_name.upper():10} | 形状: {matrix.shape} | 维度: {matrix.shape[1]}")
    
    # 可以使用特征矩阵进行后续分析
    # 例如计算距离矩阵
    print("\n可以使用这些特征矩阵计算距离矩阵了")

#%%
if __name__ == "__main__":

    all_img_paths = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Metamer_P4_C4321_Object_STI150_1300')
    all_img_paths.sort()
    all_features = main(all_img_paths)

#%%
a = np.load('resnet18_features.npy')

