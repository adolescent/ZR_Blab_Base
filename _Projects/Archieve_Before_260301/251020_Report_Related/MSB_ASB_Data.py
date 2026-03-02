'''
This script will show different graph's distance in 

'''
#%%
# import

import matplotlib.pyplot as plt
import numpy as np
import OS_Tools as ot
import seaborn as sns
import pandas as pd
from tqdm import tqdm

msb_resp = ot.Load_Variable(r'D:\_DataTemp\Metamer\ceiled_response\MSB\250902_ZhuangZhuang_Metamer_P4_C4321_Object_STI150_1300_g2_MSB_PSTH_Ceiled.pkl')[0]
asb_resp = ot.Load_Variable(r'D:\_DataTemp\Metamer\ceiled_response\ASB\250828_JianJian_Metamer_Pool4_C4321_Object_1k+FOB_g6_AL_ASB_PSTH_Ceiled.pkl')[0]

# msb_index = np.where(msb_resp[0]>0.5)[0]
# asb_index = np.where(asb_resp[0]>0.5)[0]
# msb_resp = msb_resp[1][msb_index,:]
# asb_resp = asb_resp[1][asb_index,:]
#%% Scatter example plot
fig,ax = plt.subplots(figsize=(3,5),dpi=240,ncols=1,nrows=3)
sns.heatmap(msb_resp[:,2,:],center=0,xticklabels=False,yticklabels=False,cbar=False,vmax=0.2,ax=ax[0])
sns.heatmap(msb_resp[:,322,:],center=0,xticklabels=False,yticklabels=False,cbar=False,vmax=0.2,ax=ax[1])
sns.heatmap(msb_resp[:,922,:],center=0,xticklabels=False,yticklabels=False,cbar=False,vmax=0.2,ax=ax[2])
for i in range(3):
    ax[i].plot([100,100],[0,227],color='yellow',alpha=0.5)
fig.tight_layout()
#%% Getting response redplot.
from Spike_Tools import *
asb_resp = asb_resp[:,:,160:320].sum(-1)
# asb_resp_avr=Redplot(asb_resp)
#%%
msb_avr = msb_resp[:,:1000,:].reshape((227,1000,45,10)).mean(-1)[:,[2,322,922],:]
asb_avr = asb_resp[:,72:,:].reshape((354,1000,45,10)).mean(-1)[:,[2,322,922],:]
# asb_avr = asb_resp[112:,72:,:].reshape((242,1000,45,10)).mean(-1)[:,[2,322,922],:]


all_resp = pd.DataFrame(columns=['Resp','Cell','Fig','Time','Area'],index = range(100000))
counter = 0
for i in tqdm(range(227)):
    for j in range(3):
        for k in range(45):
            all_resp.iloc[counter,:] = [msb_avr[i,j,k],i,j,k,'MSB']
            counter+=1

for i in tqdm(range(354)):
# for i in tqdm(range(242)):
    for j in range(3):
        for k in range(45):
            all_resp.iloc[counter,:] = [asb_avr[i,j,k],i,j,k,'ASB']
            counter+=1
all_resp = all_resp.dropna(how='any')
all_resp['Resp']=all_resp['Resp'].astype('f8')
all_resp['Time']=all_resp['Time'].astype('f8')
#%% Plot parts
plotable = all_resp.groupby('Area').get_group('ASB')
# plotable = plotable.groupby('Cell').get_group(0)
fig,ax = plt.subplots(figsize=(5,3.5),dpi=240,ncols=1,nrows=1)
# ax.plot(msb_avr[2,:])
# ax.plot(msb_avr[82,:])
# ax.plot(msb_avr[162,:])
sns.lineplot(data = plotable,y='Resp',x='Time',hue='Fig',ax=ax,palette='tab10',legend=False,errorbar=('ci',0))
ax.axvline(x=10,linestyle='--',color='gray')
# ax.legend()
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('')
ax.set_xlabel('')

#%%
####################### Part2, dist  based on sum response###################

asb_resp = ot.Load_Variable(r'D:\_DataTemp\Metamer\ceiled_response\ASB\Sorted_Response.pkl')
msb_resp = ot.Load_Variable(r'D:\_DataTemp\Metamer\ceiled_response\MSB\Sorted_Response.pkl')
msb_index = np.where(msb_resp[0]>0.5)[0]
asb_index = np.where(asb_resp[0]>0.5)[0]

asb_resp = asb_resp[1][asb_index,:]
msb_resp = msb_resp[1][msb_index,:]
# asb_resp = asb_resp[2][asb_index,:,150:250].sum(-1)
# msb_resp = msb_resp[2][msb_index,:,150:250].sum(-1)
#%%
from scipy.stats import pearsonr
a_vec = msb_resp[:,2]
b_vec = msb_resp[:,322]
c_vec = msb_resp[:,922]
# a_vec = msb_resp[:,2]
# b_vec = msb_resp[:,322]
# c_vec = msb_resp[:,922]
ab_corr,_ = pearsonr(a_vec,b_vec)
bc_corr,_ = pearsonr(b_vec,c_vec)
ac_corr,_ = pearsonr(a_vec,c_vec)
ab_dist = 1-ab_corr
bc_dist = 1-bc_corr
ac_dist = 1-ac_corr
#
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

all_img_path = ot.Get_File_Name(r'D:\_DataTemp\#stimuli\tmp2')
vertices = create_triangle_with_images(all_img_path, [ab_dist,ac_dist,bc_dist])
print("顶点坐标:", vertices)

