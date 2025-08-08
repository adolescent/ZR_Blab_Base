'''
Put stim into alexnet, getting it's response.
Then extact response of specific layers.

First work on DCNN, so be patient.

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
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



#%%
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()

with open('imagenet_class_index.json', 'r', encoding='utf-8') as file:
    class_idx = json.load(file)

#
# Download an example image from the pytorch website
# import urllib
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

#%%
# sample execution (requires torchvision)
from PIL import Image,ImageOps
from torchvision import transforms
# input_image = Image.open(filename)
# input_image = Image.open('dog4.jpg')
input_image = Image.open(r'D:\#Data\#stimuli\silct\silct_npx_1416\1400.jpg').convert('L').convert('RGB')
# input_image = Image.open(r'D:\#Data\#stimuli\silct\silct_npx_1416\1257.jpg')
# input_image = ImageOps.invert(input_image)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

## register hook

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

# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)
plt.plot(probabilities.cpu().numpy())
# print(class_idx[str(probabilities.argmax().cpu().numpy())][-1])
top5_prob, top5_catid = torch.topk(probabilities, 5)
# 获取类别名称
for i in range(len(top5_catid)):
    class_id = top5_catid[i].item()
    class_name = class_idx[str(class_id)][-1]
    confidence = top5_prob[i].item()
    print(f"{i+1}: {class_name} ({confidence*100:.2f}%)")

#%%
fig,ax = plt.subplots(ncols=6,nrows=6,dpi = 300,figsize = (7,7))

for i in range(36):
    c_response = activations['conv5'].cpu().numpy()[0,i+72,:,:]
    sns.heatmap(c_response,center = 0,xticklabels=False,yticklabels=False,cbar=False,ax = ax[i//6,i%6])
    
fig.tight_layout()


