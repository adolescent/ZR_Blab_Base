'''
这个脚本用来处理文章中的结果:
比较三选一的ODD-1 detection任务,
处理类似原始文本中的三分类任务,根据神经表征挑选三分类的正确率,以及比较vgg 和alexnet fc6 进行表征区分的正确率。
------
有两种方式：
- 原始 vs 打乱
- 原始图A vs 原始图B
------
使用的是皮尔逊距离:sqrt(1-r^2),计算每个和剩下两个的平均距离然后做softmax,方法非常简单。


'''

#%%
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os 
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
from tqdm import tqdm
from scipy.stats import pearsonr


raw_path = r'E:\#Stimsets\Raw_Objects'
raw_names = ot.Get_File_Name(raw_path)


#%%
# 1. 定义图片预处理流程
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
    ])

# 3. 加载预训练模型并设为评估模式
# AlexNet 和 VGG16
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).eval().to('cuda')
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval().to('cuda')
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval().to('cuda')

def extract_resnet_features(img_tensor):
    """
    手动提取ResNet50的layer4输出（对应AlexNet的fc6位置）
    """
    with torch.no_grad():
        x = img_tensor
        
        # 通过前几层
        x = resnet50.conv1(x)
        x = resnet50.bn1(x)
        x = resnet50.relu(x)
        x = resnet50.maxpool(x)
        
        # 通过各个残差块
        x = resnet50.layer1(x)
        x = resnet50.layer2(x)
        x = resnet50.layer3(x)
        
        # 提取layer4的输出（对应AlexNet fc6）
        x = resnet50.layer4(x)  # 形状: [batch, 2048, 7, 7]
        
        # 应用全局平均池化，得到2048维特征向量
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # 展平
        
    return x


# 5. 定义特征抽取的函数
def DCNN_Response(img_path):
    img_pill = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img_pill).unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        # AlexNet：手动提取fc6（ReLU前）
        features = alexnet.features(img_tensor)
        features = torch.flatten(features, 1)
        features = alexnet.classifier[0](features)  # Dropout
        alex_fc6_before_relu = alexnet.classifier[1](features)  # Linear（ReLU前）
        
        # VGG16：手动提取fc6（ReLU前）
        features_vgg = vgg16.features(img_tensor)
        features_vgg = torch.flatten(features_vgg, 1)
        vgg_fc6_before_relu = vgg16.classifier[0](features_vgg)  # Linear（ReLU前）
        # vgg_fc6_before_relu = vgg16.features[28](features_vgg)
        
        #Resnet50: 手动提取layer4
        features_res50 = extract_resnet_features(img_tensor)

        # 同时获取ReLU后的结果用于对比
        alex_fc6_after_relu = torch.relu(alex_fc6_before_relu)
        vgg_fc6_after_relu = torch.relu(vgg_fc6_before_relu)
    
    # return (alex_fc6_before_relu.cpu().numpy().flatten(),
    #         alex_fc6_after_relu.cpu().numpy().flatten(),
    #         vgg_fc6_before_relu.cpu().numpy().flatten(),
    #         vgg_fc6_after_relu.cpu().numpy().flatten())
    return alex_fc6_before_relu.cpu().numpy().flatten(), vgg_fc6_before_relu.cpu().numpy().flatten(),features_res50.cpu().numpy().flatten()


#%%################################# Run parts #################################
# for all 
from scipy.special import softmax
from itertools import combinations
all_names = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\Metamer_P4_C4321_Object_STI150_1300')
all_names.sort()

correct_rate = pd.DataFrame(index=range(100000),columns=['Constrain','Graph','Prop_Correct','Network'])


# graph_id = 1
# c_level = 1
# data = [graph_id+40*c_level,graph_id+200+40*c_level,graph_id+400+40*c_level,graph_id+600+40*c_level,graph_id+800+40*c_level]
# all_pairs = list(combinations(data, 2))

counter=0
for l in range(1,5):
    c_level = l
    c_constrain = 5-l
    print(f'Current Constrain:C{c_constrain}')
    for j in tqdm(range(20)):
        graph_id =j
        data = [graph_id+40*c_level,graph_id+200+40*c_level,graph_id+400+40*c_level,graph_id+600+40*c_level,graph_id+800+40*c_level]
        all_pairs = list(combinations(data, 2))
        # print(f'Current ID: {graph_id}')
        for i,c_pair in enumerate(all_pairs):
            raw_alex,raw_vgg,raw_res = DCNN_Response(all_names[graph_id])
            c4_alex,c4_vgg,c4_res = DCNN_Response(all_names[c_pair[0]])
            c42_alex,c42_vgg,c42_res = DCNN_Response(all_names[c_pair[1]])
            # alexnet
            a,_ = pearsonr(raw_alex,c4_alex)
            b,_ = pearsonr(raw_alex,c42_alex)
            c,_ = pearsonr(c4_alex,c42_alex)
            # a = np.sqrt(1-a**2)
            # b = np.sqrt(1-b**2)
            # c = np.sqrt(1-c**2)
            a = 1-a
            b = 1-b
            c = 1-c
            c_correct = softmax([(a+b)/2,(a+c)/2,(b+c)/2])[0]
            correct_rate.loc[counter,:] = [c_constrain,j,c_correct,'Alexnet']
            counter += 1 
            # vgg
            a_vgg,_ = pearsonr(raw_vgg,c4_vgg)
            b_vgg,_ = pearsonr(raw_vgg,c42_vgg)
            c_vgg,_ = pearsonr(c4_vgg,c42_vgg)
            a_vgg = 1-a_vgg
            b_vgg = 1-b_vgg
            c_vgg = 1-c_vgg
            c_correct_vgg = softmax([(a_vgg+b_vgg)/2,(a_vgg+c_vgg)/2,(b_vgg+c_vgg)/2])[0]
            correct_rate.loc[counter,:] = [c_constrain,j,c_correct_vgg,'VGG16']
            counter += 1 
            # resnet
            a_res,_ = pearsonr(raw_res,c4_res)
            b_res,_ = pearsonr(raw_res,c42_res)
            c_res,_ = pearsonr(c4_res,c42_res)
            a_res = 1-a_res
            b_res = 1-b_res
            c_res = 1-c_res
            c_correct_res = softmax([(a_res+b_res)/2,(a_res+c_res)/2,(b_res+c_res)/2])[0]
            correct_rate.loc[counter,:] = [c_constrain,j,c_correct_res,'Resnet50']
            counter += 1 

correct_rate = correct_rate.dropna(how='any')
# correct_rate = correct_rate.astype('f8')
#%% Plot generated graph.

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,5),dpi=240)
correct_rate.Constrain = correct_rate.Constrain.astype('str')
sns.lineplot(data=correct_rate,x ='Constrain',y='Prop_Correct',ax=ax,errorbar='ci',hue='Network',legend=False)
sns.boxplot(data=correct_rate,x ='Constrain',y='Prop_Correct',ax=ax,hue='Network',width=0.3, showfliers=False)

ax.axhline(1/3,linestyle='--',color='gray')
ax.set_ylim(0,1)
ax.set_xticklabels([4,3,2,1])
ax.set_ylabel('Correct Prop.')

correct_rate.to_csv('Network_Correct_Rate.csv')
#%% load in real neuron data, and calculate it's odd detection effects.
neu_folder = r'E:\#Preprocessed_Data\Selected_Cells'


