'''
ai coded synthesizer.
'''
#%%
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



class FeatureSynthesizer:
    def __init__(self, target_image, spatial_constraint=(1, 1), 
                 feature_layers=['pool1', 'pool2', 'pool4'], 
                 device='cuda', learning_rate=0.05, num_steps=5000,step_mute = True):
        """
        特征匹配图像合成器
        
        参数:
        target_image: PIL.Image - 目标自然图像
        spatial_constraint: (int, int) - 空间约束 (H_blocks, W_blocks)
        feature_layers: list - 使用的VGG特征层
        device: str - 计算设备 ('cuda' 或 'cpu')
        learning_rate: float - 优化学习率
        num_steps: int - 优化迭代次数
        """
        self.device = device
        self.spatial_constraint = spatial_constraint
        self.feature_layers = feature_layers
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.step_mute = step_mute
        
        # 初始化VGG19模型
        self.vgg = models.vgg19(weights=True).features.to(self.device).eval()
        self.vgg.requires_grad_(False)
        
        # 创建层名到索引的映射
        self.layer_index_map = self._create_layer_index_map()
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 目标图像处理
        self.target_tensor = self.preprocess(target_image).unsqueeze(0).to(self.device)
        self.target_size = (224, 224)
        
        # 获取目标特征Gram矩阵
        self.target_grams = self._get_target_features()
        
        # 初始化合成图像 (随机噪声)
        self.synth_image = torch.randn_like(self.target_tensor, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.synth_image], lr=learning_rate)
        
        # 存储中间结果
        self.loss_history = []
        self.synth_history = []
    
    def _create_layer_index_map(self):
        """创建 VGG19 池化层名到索引的映射"""
        # VGG19 池化层位置 (基于标准PyTorch实现)
        return {
            'conv1_1':0,
            'conv1_2':2,
            'pool1': 4,   # 第一个池化层
            'conv2_1':5,
            'conv2_2':7,
            'pool2': 9,   # 第二个池化层
            'conv3_1':10,
            'conv3_2':12,
            'conv3_3':14,
            'conv3_4':16,
            'pool3': 18,  # 第三个池化层
            'conv4_1':19,
            'conv4_2':21, # 常用的全局conv层
            'conv4_3':23,
            'conv4_4':25,
            'pool4': 27,  # 第四个池化层
            'conv5_1':28,
            'conv5_2':30,
            'conv5_3':32,
            'conv5_4':34,
            'pool5': 36   # 第五个池化层
        }
    
    def _get_target_features(self):
        """提取目标图像的特征Gram矩阵"""
        features = {}
        x = self.target_tensor
        
        # 获取所有子模块列表
        modules = list(self.vgg.children())
        
        # 遍历所有层
        for idx, module in enumerate(modules):
            x = module(x)
            
            # 检查当前层是否是需要提取的特征层
            for layer_name, target_idx in self.layer_index_map.items():
                if idx == target_idx and layer_name in self.feature_layers:
                    # 计算空间约束的Gram矩阵列表
                    gram_list = self._compute_gram_matrix(x, self.spatial_constraint)
                    
                    # 对列表中每个Gram矩阵应用detach()
                    detached_grams = [gram.detach() for gram in gram_list]
                    features[layer_name] = detached_grams
        
        return features
    
    def _compute_gram_matrix(self, activations, spatial_constraint):
        """
        计算空间约束的Gram矩阵
        
        参数:
        activations: torch.Tensor - 特征激活图 (1, C, H, W)
        spatial_constraint: (int, int) - 区块划分 (H_blocks, W_blocks)
        
        返回:
        list - 每个区块的Gram矩阵 [block1, block2, ...]
        """
        _, C, H, W = activations.shape
        h_blocks, w_blocks = spatial_constraint
        
        # 计算每个区块的大小
        block_h = H // h_blocks
        block_w = W // w_blocks
        
        grams = []
        
        # 遍历所有区块
        for i in range(h_blocks):
            for j in range(w_blocks):
                # 提取当前区块的特征
                block = activations[:, :, 
                          i*block_h:(i+1)*block_h, 
                          j*block_w:(j+1)*block_w]
                
                # 重塑为 (C, N) 其中 N = block_h * block_w
                # block = block.view(C, -1)
                block = block.reshape(C,-1)
                
                # 计算Gram矩阵: G = block @ block.T
                # gram = torch.mm(block, block.t())
                gram = torch.mm(block.clone(), block.clone().t())
                # 归一化 (除以元素数量)
                gram = gram / (block.size(1))
                
                grams.append(gram)
        
        return grams
    
    def _compute_loss(self, activations, layer_name):
        """计算当前激活与目标Gram矩阵的损失"""
        current_grams = self._compute_gram_matrix(activations, self.spatial_constraint)
        target_grams = self.target_grams[layer_name]
        
        loss = 0
        # 对每个区块计算损失
        for cur_gram, tgt_gram in zip(current_grams, target_grams):
            # 计算Gram矩阵的均方误差
            loss += torch.mean((cur_gram - tgt_gram)**2)
        
        return loss
    
    def synthesize(self):
        """执行图像合成优化"""
        # 获取所有子模块列表
        modules = list(self.vgg.children())
        
        start_time = time.time()
        
        for step in range(self.num_steps):
            self.optimizer.zero_grad()
            x = self.synth_image
            total_loss = 0
            
            # 遍历所有层
            for idx, module in enumerate(modules):
                x = module(x)
                
                # 检查当前层是否是需要匹配的特征层
                for layer_name, target_idx in self.layer_index_map.items():
                    if idx == target_idx and layer_name in self.feature_layers:
                        layer_loss = self._compute_loss(x, layer_name)
                        weight = self._get_layer_weight(layer_name)
                        total_loss += weight * layer_loss
            
            # 反向传播和优化
            total_loss.backward()
            self.optimizer.step()
            
            # 记录损失
            self.loss_history.append(total_loss.item())
            
            # 每100步保存中间结果
            if step % 1000 == 0:
                with torch.no_grad():
                    synth_img = self._postprocess(self.synth_image)
                    self.synth_history.append((step, synth_img.copy()))
            
            # 打印进度
            if self.step_mute == False:
                if step % 1000 == 0:
                    print(f"Step {step}/{self.num_steps}, Loss: {total_loss.item():.6f}")

        if self.step_mute == False:
            print(f"合成完成! 耗时: {time.time()-start_time:.2f}秒")

            
        return self._postprocess(self.synth_image)
    
    def _get_layer_weight(self, layer_name):
        """获取不同层的损失权重"""
        weights = {
            'pool1': 1.0,
            'pool2': 2.0,
            'pool3': 3.0,
            'pool4': 5.0,
            'pool5': 3.0,
            'conv1_2':0.000001,
            'conv4_2':0.000001,
            'conv3_2':0.000001,
        }
        return weights.get(layer_name, 1.0)
    
    def _postprocess(self, tensor):
        """将张量转换为PIL图像"""
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        tensor = tensor * std + mean
        
        # 裁剪到[0,1]范围
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy数组并调整维度
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image = (image * 255).astype(np.uint8)
        
        return Image.fromarray(image)
    
    def visualize_results(self):
        """可视化合成过程和结果"""
        plt.figure(figsize=(15, 10))
        
        # 显示损失曲线
        plt.subplot(2, 3, 1)
        plt.plot(self.loss_history)
        plt.title('Loss')
        plt.xlabel('N Iter')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 显示目标图像
        plt.subplot(2, 3, 2)
        target_img = self._postprocess(self.target_tensor)
        plt.imshow(target_img)
        plt.title('Target')
        
        # 显示最终合成图像
        plt.subplot(2, 3, 3)
        synth_img = self._postprocess(self.synth_image)
        plt.imshow(synth_img)
        plt.title(f'Synth (constrain: {self.spatial_constraint})')
        
        # 显示中间合成过程

        plt.subplot(2, 3, 4)
        plt.imshow(self.synth_history[1][1])
        plt.title(f'Iter {self.synth_history[1][0]}')
        mid_num = len(self.synth_history)//2
        plt.subplot(2, 3, 5)
        plt.imshow(self.synth_history[mid_num][1])
        plt.title(f'Iter {self.synth_history[mid_num][0]}')
        plt.subplot(2, 3, 6)
        plt.imshow(self.synth_history[-1][1])
        plt.title(f'Iter {self.synth_history[-1][0]}')
        
        plt.tight_layout()
        plt.show()

#%% 使用示例
if __name__ == "__main__":
    # 加载目标图像
    target_image = Image.open("0079.jpg").convert("RGB")
    # target_image = target_image.resize((224,224))
    # target_image = Image.open(r"D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\real_stim\0004.jpg").convert("RGB")
    
    # 创建合成器 (不同空间约束)
    synthesizer_global = FeatureSynthesizer(
        target_image, 
        spatial_constraint=(4,4),  # 全局约束 (完全打乱)
        feature_layers=['pool1','pool2','pool4'],
        num_steps=15000,step_mute=False,
        learning_rate=0.05
    )
    
    
    # 执行合成
    synth_global = synthesizer_global.synthesize()
    # 可视化结果
    synthesizer_global.visualize_results()
    synthesizer_global
    # 保存结果
    synth_global.save("synth.jpg")

    

