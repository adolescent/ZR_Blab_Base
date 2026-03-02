'''

This script will show how metamer change Alexnet response, and how alexnet vary classifier.

'''

#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from PIL import Image,ImageOps
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

import os 
import Common_Functions.OS_Tools as ot

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=models.AlexNet_Weights.DEFAULT)
model.eval()

model.to('cuda')


#%%
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.image_paths.sort()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def FC_Extractor(dataloader,layer='fc6'):
    
    # activations = {}
    # def get_activation(name):
    #     """钩子函数：捕获指定层的输出"""
    #     def hook(model, input, output):
    #         activations[name] = output.detach().cpu()
    #     return hook
    activations = []
    def get_output(module, input, output):
        activations.append(output.cpu().detach())

    # vgg19.features[4].register_forward_hook(get_output('pool1'))# Maxpool 1
    # vgg19.features[9].register_forward_hook(get_output('pool2'))# Maxpool 2
    # vgg19.features[18].register_forward_hook(get_output('pool1'))# Maxpool 3
    # vgg19.features[27].register_forward_hook(get_output('pool2'))# Maxpool 4
    # vgg19.features[36].register_forward_hook(get_output('pool1'))# Maxpool 5
    # vgg19.classifier[0].register_forward_hook(get_output('fc1')) # full connection 1
    # vgg19.classifier[3].register_forward_hook(get_output('fc2')) # full connection 2
    # vgg19.features[2].register_forward_hook(get_output('conv1_2')) # conv layers 1_2
    # vgg19.features[21].register_forward_hook(get_output('conv4_2')) # conv layers 4_2
    # add more if required.
    if layer == 'fc6':
        hook = model.classifier[1].register_forward_hook(get_output) # full connection 1
    elif layer == 'fc7':
        hook = model.classifier[4].register_forward_hook(get_output) # full connection 1

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to('cuda')
            _ = model(batch)

    # extracted_response = activations[layer]
    extracted_response = torch.cat(activations, dim=0)
    hook.remove()

    return extracted_response.cpu().numpy()


#%%
if __name__ == '__main__':
    # img_path = r'D:\_DataTemp\#stimuli\tmp2'
    img_path = r'Z:\Monkey\Stimuli\ZR\Metamer_P4_C4321_Object_STI150_1300'
    all_img_path = ot.Get_File_Name(img_path)
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageDataset(img_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    fc6_resps = FC_Extractor(dataloader,'fc6')

#%% plot metamer and raw graph
# from scipy.stats import pearsonr
# fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(5,5),dpi=240,sharex=True)
# ax[0].plot(fc6_resps[2,2000:2500],color=plt.cm.tab10(0))
# ax[1].plot(fc6_resps[322,2000:2500]*1.5-70,color=plt.cm.tab10(1))
# ax[2].plot(fc6_resps[922,2000:2500]*1.5-70,color=plt.cm.tab10(2))

# ax[0].set_yticks([])
# ax[1].set_yticks([])
# ax[2].set_yticks([])
# ax[2].set_xticks([])

# fig.tight_layout()
#%% calculate distance between image pairs
from scipy.stats import pearsonr
def cosine_distance(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: # Handle zero vectors
        return 1.0 # Maximum dissimilarity
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    return 1 - cosine_similarity

# ab_dist = np.linalg.norm(fc6_resps[0,:]-fc6_resps[1,:])
# ac_dist = np.linalg.norm(fc6_resps[0,:]-fc6_resps[2,:])
# bc_dist = np.linalg.norm(fc6_resps[1,:]-fc6_resps[2,:])
# ab_dist = cosine_distance(fc6_resps[0,:],fc6_resps[1,:])
# ac_dist = cosine_distance(fc6_resps[0,:],fc6_resps[2,:])
# bc_dist = cosine_distance(fc6_resps[1,:],fc6_resps[2,:])
ab_corr,_ = pearsonr(fc6_resps[0,:],fc6_resps[1,:])
bc_corr,_ = pearsonr(fc6_resps[1,:],fc6_resps[2,:])
ac_corr,_ = pearsonr(fc6_resps[0,:],fc6_resps[2,:])
ab_dist = 1-ab_corr
bc_dist = 1-bc_corr
ac_dist = 1-ac_corr


#%% plot distance graphs.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

def create_triangle_with_images(image_paths, distances, figsize=(12, 10)):
    """
    在三角形的三个顶点放置图片，并根据给定的距离调整三角形大小
    
    参数:
    image_paths: 三张图片路径的列表 [img1_path, img2_path, img3_path]
    distances: 三个距离的列表 [d12, d13, d23]，分别对应边12、边13、边23的长度
    figsize: 图形大小
    """
    
    # 验证输入
    if len(image_paths) != 3 or len(distances) != 3:
        raise ValueError("需要恰好3张图片和3个距离")
    
    # 解包距离
    d12, d13, d23 = distances
    
    # 验证三角形不等式
    if not (d12 + d13 > d23 and d12 + d23 > d13 and d13 + d23 > d12):
        raise ValueError("给定的距离不满足三角形不等式，无法构成三角形")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算三角形顶点坐标
    # 将第一个点放在原点
    x1, y1 = 0, 0
    # 第二个点在x轴上
    x2, y2 = d12, 0
    
    # 计算第三个点的坐标（使用余弦定理）
    cos_angle = (d12**2 + d13**2 - d23**2) / (2 * d12 * d13)
    sin_angle = np.sqrt(1 - cos_angle**2)
    x3 = d13 * cos_angle
    y3 = d13 * sin_angle
    
    # 绘制三角形边
    vertices = [(x1, y1), (x2, y2), (x3, y3)]
    triangle = plt.Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # 在顶点添加图片
    for i, (x, y) in enumerate(vertices):
        # 加载图片
        img = mpimg.imread(image_paths[i])
        
        # 创建OffsetImage对象
        imagebox = OffsetImage(img, zoom=0.3)  # 调整zoom参数控制图片大小
        
        # 创建AnnotationBbox将图片放在指定位置
        ab = AnnotationBbox(imagebox, (x, y), 
                           frameon=True, 
                           pad=0.5,
                           boxcoords="data")
        ax.add_artist(ab)
        
        # 添加顶点标签（可选）
        # ax.text(x, y-0.1, f'Point {i+1}', ha='center', va='top', fontsize=12)
    
    # # 添加距离标签
    # ax.text((x1+x2)/2, (y1+y2)/2 - 0.1, f'd={d12}', ha='center', va='top', fontsize=10)
    # ax.text((x1+x3)/2, (y1+y3)/2 - 0.1, f'd={d13}', ha='center', va='top', fontsize=10)
    # ax.text((x2+x3)/2, (y2+y3)/2 - 0.1, f'd={d23}', ha='center', va='top', fontsize=10)
    
    # 设置坐标轴
    margin = max(distances) * 0.2
    # ax.set_xlim(min(x1, x2, x3) - margin, max(x1, x2, x3) + margin)
    # ax.set_ylim(min(y1, y2, y3) - margin, max(y1, y2, y3) + margin)
    ax.set_xlim((-0.1,1))
    ax.set_ylim((-0.1,1))
    ax.set_yticks([-0.1,0,1])
    ax.set_xticks([-0.1,0,1])
    
    
    ax.set_aspect('equal')
    ax.axis('off')  # 隐藏坐标轴
    
    # plt.title('Triangle with Images at Vertices', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return vertices

vertices = create_triangle_with_images(all_img_path, [ab_dist,ac_dist,bc_dist])
print("顶点坐标:", vertices)
