'''

We use alexnet's fc6 and fc7, try to generate an rsa matrix for all silct graphs.
'''



#%%
'''
import alexnet from pytorch, and 
'''
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
import OS_Tools as ot

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=models.AlexNet_Weights.DEFAULT)
model.eval()

model.to('cuda')

#%%
'''
define response extractor.
'''

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
    img_path = r'D:\#stimuli\Scramble_Global_PNAS_VGG19_50_imgnet\Pool4_C4321_Object_only_Repeat5'
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

    fc6_resps = FC_Extractor(dataloader,'fc6')
    fc7_resps = FC_Extractor(dataloader,'fc7')



    
    # re-arrange plots for 
    from Matrix_Tools import *
    fc6_corr = Corr_Matrix(data = fc6_resps.T,fill_diag=True)
    fc7_corr = Corr_Matrix(data = fc7_resps.T,fill_diag=True)
    n = 1000
    # Raw-C4-C3-C2-C1,with 5 repeat
    new_ids = []
    for i in range(40):
        new_ids.extend(list(np.arange(i, n, 40)))  # 0, 3, 6, 9, ...
        # indices_1 = np.arange(1, n, 40)  # 1, 4, 7, 10, ...
        # indices_2 = np.arange(2, n, 40)  # 2, 5, 8, 11, ...
        # indices_3 = np.arange(3, n, 40)  # 1, 4, 7, 10, ...
        # indices_4 = np.arange(4, n, 40)  # 2, 5, 8, 11, ...
    # 合并索引
    # new_indices = np.concatenate(new_ids)
    sorted_fc6 = fc6_corr[new_ids, :][:, new_ids] # ignore fob
    sorted_fc7 = fc7_corr[new_ids, :][:, new_ids] # ignore fob
    

    # sns.heatmap(sorted_fc1,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False)
    # sns.heatmap(sorted_fc2,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False)
    fig,ax = plt.subplots(figsize = (8,4),dpi=300,ncols=2,nrows=1)
    sns.heatmap(sorted_fc6,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False,ax = ax[0],vmax=0.85,vmin=0)
    sns.heatmap(sorted_fc7,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False,ax = ax[1],vmax=0.85,vmin=0)
    fig.tight_layout()


