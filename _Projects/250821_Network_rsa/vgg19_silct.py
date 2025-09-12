'''
Almost the same work as alexnet version, but used on vgg19.
'''


#%%


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from PIL import Image,ImageOps
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import os 
import OS_Tools as ot

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT, progress=True)
vgg19.to('cuda')
#%%
# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def FC_Extractor(dataloader,layer='fc1'):
    


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
    if layer == 'fc1':
        hook = vgg19.classifier[0].register_forward_hook(get_output) # full connection 1
    elif layer == 'fc2':
        hook = vgg19.classifier[3].register_forward_hook(get_output) # full connection 1
    

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to('cuda')
            _ = vgg19(batch)

    # extracted_response = activations[layer]
    extracted_response = torch.cat(activations, dim=0)
    hook.remove()

    return extracted_response.cpu().numpy()
    



#%% test run part
if __name__ == '__main__':

    img_path = r'D:\#stimuli\silct\silct_npx_1416'
    all_img_path = ot.Get_File_Name(img_path)
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageDataset(img_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    fc1_resps = FC_Extractor(dataloader,'fc1')
    fc2_resps = FC_Extractor(dataloader,'fc2')
    

    # input_image = Image.open(all_img_path[153])
    # fc1_resps = np.zeros(shape=(len(all_img_path),4096))
    # fc2_resps = np.zeros(shape=(len(all_img_path),4096))   
    # # a = FC_Extractor(input_image,'fc6')
    # # get fc response of all imgs, and getting it's rsa matrix.
    # for i,c_img_path in tqdm(enumerate(all_img_path)):
    #     c_img = Image.open(c_img_path)
    #     c1_resp = FC_Extractor(c_img,'fc1')
    #     c2_resp = FC_Extractor(c_img,'fc2')
    #     fc1_resps[i,:] = c1_resp[0,:]
    #     fc2_resps[i,:] = c2_resp[0,:]
    from Matrix_Tools import *
    fc1_corr = Corr_Matrix(data = fc1_resps.T,fill_diag=True)
    fc2_corr = Corr_Matrix(data = fc2_resps.T,fill_diag=True)
    n = 1200
    # 先取所有索引中模3余0的，然后模3余1的，最后模3余2的
    indices_0 = np.arange(0, n, 3)  # 0, 3, 6, 9, ...
    indices_1 = np.arange(1, n, 3)  # 1, 4, 7, 10, ...
    indices_2 = np.arange(2, n, 3)  # 2, 5, 8, 11, ...
    # 合并索引
    new_indices = np.concatenate((indices_0, indices_1, indices_2))
    sorted_fc1 = fc1_corr[new_indices, :][:, new_indices] # ignore fob
    sorted_fc2 = fc2_corr[new_indices, :][:, new_indices] # ignore fob
    

    # sns.heatmap(sorted_fc1,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False)
    # sns.heatmap(sorted_fc2,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False)
    fig,ax = plt.subplots(figsize = (8,4),dpi=300,ncols=2,nrows=1)
    sns.heatmap(sorted_fc1,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False,ax = ax[0],vmax=0.85,vmin=0)
    sns.heatmap(sorted_fc2,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False,ax = ax[1],vmax=0.85,vmin=0)
    fig.tight_layout()


