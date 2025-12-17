'''
This script will show perform of SVM on raw 20 class evaluation

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


wp=r'E:\#Preprocessed_Data\Selected_Cells'
infos,_,_ = Load_Info('Mega_Metamer_v250920')
infos = infos.iloc[300:,:].reset_index(drop=True)

#%% load in data
acs_msb = np.load(ot.Join(wp,'MSB_Cells_Metamer_Cutshuffle.npz'),allow_pickle=True)
msb_resps = acs_msb['psth']
msb_resps = msb_resps[:,:,160:320].sum(-1) # use 60-220ms response
msb_cell_dps = pd.DataFrame(acs_msb['d_primes'],columns=['Cell','D_Prime','Category','Loc','Cell_ID'])
msb_cell_resps = pd.DataFrame(acs_msb['response'],columns=['Cell','Category','Response','Loc','Cell_ID'])

acs_asb = np.load(ot.Join(wp,'ASB_Cells_Metamer_Cutshuffle.npz'),allow_pickle=True)
asb_resps = acs_asb['psth']
asb_resps = asb_resps[:,:,160:320].sum(-1) # use 60-220ms response
asb_cell_dps = pd.DataFrame(acs_asb['d_primes'],columns=['Cell','D_Prime','Category','Loc','Cell_ID'])
asb_cell_resps = pd.DataFrame(acs_asb['response'],columns=['Cell','Category','Response','Loc','Cell_ID'])

#%%####################### SVM Trainer ########################
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

## getting all animate response of MSB, and it's label here.
ani_raws = infos[infos['Category'].str.contains('Raw_Ani', na=False)]
labels = np.array(ani_raws['Object'])
raw_rsps = msb_resps[:,np.array(ani_raws.index)]

## train svm classifier on raw response
X = raw_rsps.T
y=labels
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

stim_info = infos
ani_c4 = stim_info[stim_info['Category'].str.contains('P4_C4_Ani', na=False)]
labels_c4 = np.array(ani_c4['Object'])
c4_rsps = msb_resps[:,np.array(ani_c4.index)]
ani_c3 = stim_info[stim_info['Category'].str.contains('P4_C3_Ani', na=False)]
labels_c3 = np.array(ani_c3['Object'])
c3_rsps = msb_resps[:,np.array(ani_c3.index)]
ani_c2 = stim_info[stim_info['Category'].str.contains('P4_C2_Ani', na=False)]
labels_c2 = np.array(ani_c2['Object'])
c2_rsps = msb_resps[:,np.array(ani_c2.index)]
ani_c1 = stim_info[stim_info['Category'].str.contains('P4_C1_Ani', na=False)]
labels_c1 = np.array(ani_c1['Object'])
c1_rsps = msb_resps[:,np.array(ani_c1.index)]

c4_acc,c4_pred = test_svm_with_response_data(svm_classifier,c4_rsps,labels_c4)
c3_acc,c3_pred = test_svm_with_response_data(svm_classifier,c3_rsps,labels_c3)
c2_acc,c2_pred = test_svm_with_response_data(svm_classifier,c2_rsps,labels_c2)
c1_acc,c1_pred = test_svm_with_response_data(svm_classifier,c1_rsps,labels_c1)
