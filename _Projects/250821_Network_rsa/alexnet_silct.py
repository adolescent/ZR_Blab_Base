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

import os 
import OS_Tools as ot

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()



#%%
'''
define response extractor.
'''
def FC_Extractor(img,layer='fc6'):

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    


    activations = {}
    def get_activation(name):
        """钩子函数：捕获指定层的输出"""
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    # 为各层注册钩子
    model.features[0].register_forward_hook(get_activation('conv1'))      # 第1卷积层
    model.features[3].register_forward_hook(get_activation('conv2'))      # 第2卷积层
    model.features[6].register_forward_hook(get_activation('conv3'))      # 第3卷积层
    model.features[8].register_forward_hook(get_activation('conv4'))      # 第4卷积层
    model.features[10].register_forward_hook(get_activation('conv5'))     # 第5卷积层
    model.classifier[1].register_forward_hook(get_activation('fc6'))      # FC6层
    model.classifier[4].register_forward_hook(get_activation('fc7'))      # FC7层
    model.classifier[6].register_forward_hook(get_activation('fc8'))      # FC8层
    with torch.no_grad():
        output = model(input_batch)

    extracted_response = activations[layer]

    return extracted_response.cpu().numpy()


#%%

if __name__ == '__main__':
    img_path = r'D:\#stimuli\silct\silct_npx_1416'
    all_img_path = ot.Get_File_Name(img_path)

    # input_image = Image.open(all_img_path[153])
    fc6_resps = np.zeros(shape=(len(all_img_path),4096))
    fc7_resps = np.zeros(shape=(len(all_img_path),4096))   
    # a = FC_Extractor(input_image,'fc6')
    # get fc response of all imgs, and getting it's rsa matrix.
    for i,c_img_path in tqdm(enumerate(all_img_path)):
        c_img = Image.open(c_img_path)
        c6_resp = FC_Extractor(c_img,'fc6')
        c7_resp = FC_Extractor(c_img,'fc7')
        fc6_resps[i,:] = c6_resp[0,:]
        fc7_resps[i,:] = c7_resp[0,:]

    from Matrix_Tools import *
    fc6_corr = Corr_Matrix(data = fc6_resps.T,fill_diag=True)
    fc7_corr = Corr_Matrix(data = fc7_resps.T,fill_diag=True)
    
    # re-arrange plots for 
    n = 1200
    # 先取所有索引中模3余0的，然后模3余1的，最后模3余2的
    indices_0 = np.arange(0, n, 3)  # 0, 3, 6, 9, ...
    indices_1 = np.arange(1, n, 3)  # 1, 4, 7, 10, ...
    indices_2 = np.arange(2, n, 3)  # 2, 5, 8, 11, ...
    # 合并索引
    new_indices = np.concatenate((indices_0, indices_1, indices_2))
    sorted_fc6 = fc6_corr[new_indices, :][:, new_indices] # ignore fob
    sorted_fc7 = fc7_corr[new_indices, :][:, new_indices] # ignore fob
    

    sns.heatmap(sorted_fc6,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False)
    sns.heatmap(sorted_fc7,center=0,square=True,xticklabels=False,yticklabels=False,cbar=False)
    

    