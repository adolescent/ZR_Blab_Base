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
import OS_Tools as ot

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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    fc6_resps = FC_Extractor(dataloader,'fc6')

#%% plot metamer and raw graph
from scipy.stats import pearsonr
fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(5,3),dpi=240,sharex=True)
ax[0].plot(fc6_resps[2,2000:2500],color=plt.cm.tab10(0))
ax[1].plot(fc6_resps[162,2000:2500]*1.5-70,color=plt.cm.tab10(1))
ax[0].set_yticks([])
ax[1].set_yticks([])
ax[0].set_xticks([])

fig.tight_layout()

# pearsonr(fc6_resps[2,:],fc6_resps[162,:])

