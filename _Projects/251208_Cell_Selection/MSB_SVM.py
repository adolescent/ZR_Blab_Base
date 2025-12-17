'''
Do an SVM classifier based on MSB data, try to locate cell's ability of descrimination.

40 classifier for matamer, and 5 classifier for unique stim selection(Which type of stim can be classified?)

Raw Response classifier, Z score classifier and PCA classifier?
'''


#%%

from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
import copy
from Matrix_Tools import *

wp = r'E:\#Preprocessed_Data\Selected_Cells'

#%% load in data
acs = np.load(ot.Join(wp,'MSB_Cells.npz'),allow_pickle=True)
all_resps = acs['psth']
avr_resps = all_resps[:,:,160:320].sum(-1)
cell_dps = pd.DataFrame(acs['d_primes'],columns=['Cell','D_Prime','Category','Loc','Cell_ID'])
cell_resps = pd.DataFrame(acs['response'],columns=['Cell','Category','Response','Loc','Cell_ID'])

#%%
'''
This part will generate Redplot of Z-scored all MSB body cell.
'''
zs = avr_resps
# zs = (avr_resps-avr_resps.mean(1,keepdims=1))/avr_resps.std(1,keepdims=1)
zs = zs.reshape(len(zs),5,200).mean(1)
redplot,ids = Redplot_PCA_Arranger(zs,reverse=True)

fig,ax = plt.subplots(nrows=1,ncols=1,dpi=300,figsize=(5,6))
sns.heatmap(redplot,center=0,cmap='bwr',vmax=7,cbar=False,ax=ax,xticklabels=False,yticklabels=False)
for i,cx in enumerate([40,80,120,160]):
    ax.plot([cx,cx],[0,len(zs)],c='black',lw=1)

ax.set_yticks([0,400,800,1200])
ax.set_yticklabels([0,400,800,1200],rotation=90,size=8)
# ax.set_yticks([0,1000,2000,3000,4000])
# ax.set_yticklabels([0,1000,2000,3000,4000],rotation=90,size=8)
fig.tight_layout()

#%% getting response matrix for cells. This will show difference of rep between cells.
ranges = [(0, 20), (40, 60), (80, 100), (120, 140), (160, 180)]
# ranges = [(20,40), (60,80), (100,120), (140,160), (180,200)]
ani_parts = np.concatenate([np.arange(start, end) for start, end in ranges])
ani_rsps = zs[:,ani_parts]
corr_ani = Corr_Matrix(data=ani_rsps,fill_diag=False)
sns.heatmap(corr_ani,vmax=1,cmap='bwr',cbar=False,xticklabels=False,yticklabels=False,square=True)

#%% annotate response for 
'''
This part wil do SVM for given data, especially for
'''
# prepare training data
stim_info,_,_ = Load_Info('Metamer1300')
stim_info = stim_info.iloc[:1000,:]
ani_raws = stim_info[stim_info['Category'].str.contains('Raw_Ani', na=False)]
labels = np.array(ani_raws['Object'])
raw_rsps = avr_resps[:,np.array(ani_raws.index)]

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

X = raw_rsps.T
y=labels
# svm training
svm_pipeline = make_pipeline(
    StandardScaler(),  # 标准化特征
    # SVC(kernel='linear', C=1.0, random_state=42)  # 线性SVM
    SVC(kernel='linear', C=1.0)
)

# 使用分层5折交叉验证（保持类别比例）
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 进行交叉验证
cv_scores = cross_val_score(
    svm_pipeline, 
    X, 
    y, 
    cv=stratified_kfold,
    scoring='accuracy'
)

print(f"5折交叉验证准确率: {cv_scores}")
print(f"平均准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 在整个数据集上训练最终的SVM分类器
svm_pipeline.fit(X, y)

# 返回分类器和准确率
svm_classifier = svm_pipeline
accuracy = cv_scores.mean()

print(f"\n训练完成！")
print(f"最终分类器: {svm_classifier}")
print(f"平均准确率: {accuracy:.4f}")

# 测试分类器在训练集上的性能
train_accuracy = svm_classifier.score(X, y)
print(f"训练集准确率: {train_accuracy:.4f}")

# prepare data and test dataset.
ani_c4 = stim_info[stim_info['Category'].str.contains('P4_C4_Ani', na=False)]
labels_c4 = np.array(ani_c4['Object'])
c4_rsps = avr_resps[:,np.array(ani_c4.index)]
ani_c3 = stim_info[stim_info['Category'].str.contains('P4_C3_Ani', na=False)]
labels_c3 = np.array(ani_c3['Object'])
c3_rsps = avr_resps[:,np.array(ani_c3.index)]
ani_c2 = stim_info[stim_info['Category'].str.contains('P4_C2_Ani', na=False)]
labels_c2 = np.array(ani_c2['Object'])
c2_rsps = avr_resps[:,np.array(ani_c2.index)]
ani_c1 = stim_info[stim_info['Category'].str.contains('P4_C1_Ani', na=False)]
labels_c1 = np.array(ani_c1['Object'])
c1_rsps = avr_resps[:,np.array(ani_c1.index)]

def test_svm_with_response_data(svm_classifier, test_responses, test_labels):
    """
    针对你的响应数据格式进行测试
    test_responses: (1211, n) 的numpy数组，n个测试样本
    test_labels: (n,) 的numpy数组，测试标签
    """
    # 1. 转置数据，使样本在行上，特征在列上
    # (1211, n) -> (n, 1211)
    X_test = test_responses.T
    
    # 2. 验证数据形状
    print(f"测试数据形状: X_test={X_test.shape}, y_test={test_labels.shape}")
    print(f"样本数: {X_test.shape[0]}, 特征数: {X_test.shape[1]}")
    
    # 3. 使用分类器进行预测
    y_pred = svm_classifier.predict(X_test)
    
    # 4. 计算准确率
    accuracy = accuracy_score(test_labels, y_pred)
    
    # 5. 输出结果
    # print(f"\n测试结果:")
    # print(f"  测试样本数: {len(test_labels)}")
    # print(f"  正确分类数: {np.sum(y_pred == test_labels)}")
    print(f"  测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 6. 返回结果
    return accuracy, y_pred

c4_acc,c4_pred = test_svm_with_response_data(svm_classifier,c4_rsps,labels_c4)
c3_acc,c3_pred = test_svm_with_response_data(svm_classifier,c3_rsps,labels_c3)
c2_acc,c2_pred = test_svm_with_response_data(svm_classifier,c2_rsps,labels_c2)
c1_acc,c1_pred = test_svm_with_response_data(svm_classifier,c1_rsps,labels_c1)
#%%
plt.plot([1,2,3,4,5],[1,c4_acc,c3_acc,c2_acc,c1_acc])
np.save('msb_svm.npy',np.array([1,c4_acc,c3_acc,c2_acc,c1_acc]))

#%% corr between 5 classes of all 40 classes.
corrs = np.zeros(shape = (40,5,5))

for k in range(40):
    for i in range(5):
        a_parts = zs[:,k*i]
        for j in range(5):
            b_parts = zs[:,k*j]
            r,_ = stats.pearsonr(a_parts,b_parts)
            corrs[k,i,j]=r
            corrs[k,j,i]=r
            
sns.heatmap(corrs[:20,:,:].mean(0),cmap='bwr',center=0.65,vmax=0.8,cbar=False,square=True)

#%%
sns.heatmap(all_resps.mean(0).reshape((5,200,450)).mean(0),center=0,cbar=False,yticklabels=False)
plt.xticks([0,100,200,300,400])
# plt.x([-100,0,100,200,300])