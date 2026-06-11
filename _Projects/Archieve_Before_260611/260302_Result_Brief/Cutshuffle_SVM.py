'''
Process svm prediction on cut shuffle case.
'''

#%%

import numpy as np
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

wp=r'E:\#Preprocessed_Data\Selected_Cells'
cellname = 'Doodle_All_ASB_Response.npz'
fullpath = ot.Join(wp,cellname)

ac_infos = np.load(fullpath,allow_pickle=True)
# ac_dp = pd.DataFrame(ac_infos['d_primes'],columns = ['Cell','DP','Category','Loc','ID'])
# ac_rsp = pd.DataFrame(ac_infos['response'],columns = ['Cell','DP','Category','Loc','ID'])
avr_rsp = ac_infos['psth'][:,:,160:320].sum(-1)

#%% get raw indexes.
stim_seq = Load_Info('Mega_Metamer_v250920')[0].loc[300:].reset_index(drop=True)
# stim_seq = Load_Info('Metamer1072')[0].loc[72:].reset_index(drop=True)
raw_ids = []
obj_index = []
for i in range(len(stim_seq)):
    cc_name = stim_seq.loc[i,'Category']
    if 'Raw' in cc_name:
        raw_ids.append(i)
        obj_index.append(stim_seq.loc[i,'Object'])
train_sets = avr_rsp[:,raw_ids]
obj_index = np.array(obj_index)
#%% generate test datasets.
p4_ids = [] 
p3_ids = []
p2_ids = []
p1_ids = []
for i,c_p in enumerate([4,3,2,1]):
    cc_tag = f'P4_C{c_p}'
    for j in range(len(stim_seq)):
        cc_name = stim_seq.loc[j,'Category']
        if cc_tag in cc_name:
            if c_p ==4:
                p4_ids.append(j)
            elif c_p ==3:
                p3_ids.append(j)
            elif c_p ==2:
                p2_ids.append(j)
            elif c_p ==1:
                p1_ids.append(j)

p4_sets = avr_rsp[:,p4_ids]
p3_sets = avr_rsp[:,p3_ids]
p2_sets = avr_rsp[:,p2_ids]
p1_sets = avr_rsp[:,p1_ids]
#%% generate cut_shuffle sets
cut12_c1 = []
cut12_c1_labels = []
cut12_c2 = []
cut12_c2_labels = []
cut12_c3 = []
cut12_c3_labels = []
cut12_c4 = []
cut12_c4_labels = []

for i,c_p in enumerate([4,3,2,1]):
    cc_tag = f'Cut12_C{c_p}'
    for j in range(len(stim_seq)):
        cc_name = stim_seq.loc[j,'Category']
        if cc_tag in cc_name:
            if c_p ==4:
                cut12_c4.append(j)
                cut12_c4_labels.append(stim_seq.loc[j,'Object'])
            elif c_p ==3:
                cut12_c3.append(j)
                cut12_c3_labels.append(stim_seq.loc[j,'Object'])
            elif c_p ==2:
                cut12_c2.append(j)
                cut12_c2_labels.append(stim_seq.loc[j,'Object'])
            elif c_p ==1:
                cut12_c1.append(j)
                cut12_c1_labels.append(stim_seq.loc[j,'Object'])

cut12_c4_labels = np.array(cut12_c4_labels)
cut12_c3_labels = np.array(cut12_c3_labels)
cut12_c2_labels = np.array(cut12_c2_labels)
cut12_c1_labels = np.array(cut12_c1_labels)
c12_p4_sets = avr_rsp[:,cut12_c4]
c12_p3_sets = avr_rsp[:,cut12_c3]
c12_p2_sets = avr_rsp[:,cut12_c2]
c12_p1_sets = avr_rsp[:,cut12_c1]
#%% c8-c9
cut8_c1 = []
cut8_c1_labels = []
cut8_c2 = []
cut8_c2_labels = []
cut8_c3 = []
cut8_c3_labels = []
cut8_c4 = []
cut8_c4_labels = []

for i,c_p in enumerate([4,3,2,1]):
    cc_tag = f'Cut8_C{c_p}'
    cc_tag2 = f'Cut9_C{c_p}'
    for j in range(len(stim_seq)):
        cc_name = stim_seq.loc[j,'Category']
        if (cc_tag in cc_name) or (cc_tag2 in cc_name):
            if c_p ==4:
                cut8_c4.append(j)
                cut8_c4_labels.append(stim_seq.loc[j,'Object'])
            elif c_p ==3:
                cut8_c3.append(j)
                cut8_c3_labels.append(stim_seq.loc[j,'Object'])
            elif c_p ==2:
                cut8_c2.append(j)
                cut8_c2_labels.append(stim_seq.loc[j,'Object'])
            elif c_p ==1:
                cut8_c1.append(j)
                cut8_c1_labels.append(stim_seq.loc[j,'Object'])

cut8_c4_labels = np.array(cut8_c4_labels)
cut8_c3_labels = np.array(cut8_c3_labels)
cut8_c2_labels = np.array(cut8_c2_labels)
cut8_c1_labels = np.array(cut8_c1_labels)
c8_p4_sets = avr_rsp[:,cut8_c4]
c8_p3_sets = avr_rsp[:,cut8_c3]
c8_p2_sets = avr_rsp[:,cut8_c2]
c8_p1_sets = avr_rsp[:,cut8_c1]
#%% C4
#%% c8-c9
cut4_c1 = []
cut4_c1_labels = []
cut4_c2 = []
cut4_c2_labels = []


for i,c_p in enumerate([4,3,2,1]):
    cc_tag = f'Cut4_C{c_p}'
    for j in range(len(stim_seq)):
        cc_name = stim_seq.loc[j,'Category']
        if (cc_tag in cc_name) or (cc_tag2 in cc_name):
            if c_p ==2:
                cut4_c2.append(j)
                cut4_c2_labels.append(stim_seq.loc[j,'Object'])
            elif c_p ==1:
                cut4_c1.append(j)
                cut4_c1_labels.append(stim_seq.loc[j,'Object'])



cut4_c2_labels = np.array(cut4_c2_labels)
cut4_c1_labels = np.array(cut4_c1_labels)

c4_p2_sets = avr_rsp[:,cut4_c2]
c4_p1_sets = avr_rsp[:,cut4_c1]



#%% train svm
## train svm classifier on raw response
X = train_sets.T
y = obj_index
svm_pipeline = make_pipeline(
    StandardScaler(),  # 标准化特征
    # SVC(kernel='linear', C=1.0, random_state=42)  # 线性SVM
    SVC(kernel='linear', C=1.0)
)
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
# 在数据集上训练最终的SVM分类器
svm_pipeline.fit(X, y)
# 返回分类器和准确率
svm_classifier = svm_pipeline
accuracy = cv_scores.mean()
print(f"\n训练完成！")
print(f"最终分类器: {svm_classifier}")
print(f"平均准确率: {accuracy:.4f}")

#%% test svm on each cut, 


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


c4_acc,c4_pred = test_svm_with_response_data(svm_classifier,c8_p4_sets,cut8_c4_labels)
c3_acc,c3_pred = test_svm_with_response_data(svm_classifier,c8_p3_sets,cut8_c3_labels)
c2_acc,c2_pred = test_svm_with_response_data(svm_classifier,c8_p2_sets,cut8_c2_labels)
c1_acc,c1_pred = test_svm_with_response_data(svm_classifier,c8_p1_sets,cut8_c1_labels)

#%% plot acc rate.
msb_acc = [0.97,0.785,0.61,0.22]
asb_acc = [0.965,0.71,0.435,0.215]
al_acc = [0.99,0.78,0.435,0.235]
ml_acc = [0.89,0.755,0.415,0.195]
c12_acc = [0.075,0.05,0.0875,0.025]
c8_acc = [0.21,0.075,0.05,0.025]
c4_acc = [0.35,0.1]

#%% plot parts
import matplotlib.pyplot as plt
fig,ax = plt.subplots(ncols=1,nrows=1,dpi = (240),figsize = (4,3))
ax.plot([4,3,2,1],asb_acc)
ax.plot([4,3,2,1],c12_acc)
ax.plot([4,3,2,1],c8_acc)
ax.plot([2,1],c4_acc)

ax.axhline(y=0.05,linestyle ='--',color='gray')

ax.set_ylim(0,1)
ax.set_xticks([4,3,2,1])
plt.gca().invert_xaxis()

#%% plot cut-shuffle
fig,ax = plt.subplots(ncols=1,nrows=1,dpi = (240),figsize = (4,3))
plotable = avr_rsp[:,1540:]
sns.heatmap(plotable/plotable.max(1,keepdims=True),center=0,cmap='bwr',ax =ax,yticklabels=False,cbar=False)
ax.set_xticks(np.arange(0,440,40))