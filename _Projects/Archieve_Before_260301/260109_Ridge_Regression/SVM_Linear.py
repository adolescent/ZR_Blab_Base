'''
使用简单的线性kernel svm分类器

1 - 用raw训练，测试对不同constrain的识别能力
2 - 用c1训练，测试泛化能力
3 - 更泛化的训练和测试方式，分块分别test&exam，判断正确率（top1，top5）
'''


#%%

#%%
# from Cell_Class import Cell_Infos
from Py_Structure.Info_Files.InfoLoader import Load_Info
import OS_Tools as ot
from Spike_Tools import *
import joblib as JL
# from joblib import dump, load
from Py_Structure.Struct_Funcs import Single_Recording_Site
from Common_Functions.Useful_Plotter import *
from tqdm import tqdm

wp=r'E:\#Preprocessed_Data\Selected_Cells'

result_path = r'E:\#Coding_traces\260109_Decoder'

msb_infos = np.load(ot.Join(wp,'MSB_Cells_Metamer_Only.npz'),allow_pickle=True)
msb_resps = msb_infos['psth']
n_msb = msb_resps.shape[0]
temp_data = msb_resps.reshape(n_msb,25,40, 450)
msb_ani_only = temp_data[:, :, :20, :].reshape(n_msb, -1, 450)
msb_ani_only_avr = msb_ani_only[:,:,160:320].sum(-1)
#%%################ 建立svm分类器 ####################
#%% 数据的标准化对结果很重要，因此使用
response_z = (msb_ani_only_avr-msb_ani_only_avr.mean(1,keepdims=True))/msb_ani_only_avr.std(1,keepdims=True)
# 对数据进行clip以避免个别神经元的强烈发放产生重大影响。
response_z = np.clip(response_z,-10,10)
##  也可也考虑用最大值norm的方式
# response_z = msb_ani_only_avr/msb_ani_only_avr.max(1,keepdims=True)
## plot 展示分布
# plt.hist(response_z.flatten(),bins=30)
#%% 根据constrain对数据进行分组，每组的label顺序都一样，
response_z_reshape = response_z.reshape((n_msb,5,100))
# ids = np.tile(np.arange(100),5) # 对应100种处理。
ids = np.tile(np.arange(20),5)
raw_rsp = response_z_reshape[:,:,:20].reshape((n_msb,-1))
c4 = response_z_reshape[:,:,20:40].reshape((n_msb,-1))
c3 = response_z_reshape[:,:,40:60].reshape((n_msb,-1))
c2 = response_z_reshape[:,:,60:80].reshape((n_msb,-1))
c1 = response_z_reshape[:,:,80:].reshape((n_msb,-1))


#%% 进行SVM分类器的训练，使用线性kernel和one hot编码
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

X_train = raw_rsp.T
X_test = c4.T

# 训练标签
y_train = ids
y_test = ids

# 数据标准化,已经使用了zscore，所以这一步省略。




