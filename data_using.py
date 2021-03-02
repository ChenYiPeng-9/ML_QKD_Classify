#============================导入外部所需训练的数据并对数据进有关行处理=========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
#==================导入数据=======================
my_data = pd.read_csv('data_finsh.csv',index_col=0)
# print(my_data.head())
# print(my_data.info)
data = my_data.values
# print(data)
# print(data.shape)
#======对特征数据标准化======
X_data = data[:,:6]
# print(X_data)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # 实例化
scaler.fit(X_data)  # fit，本质是生成均值和方差
feature_X = scaler.transform(X_data)  # 通过接口导出结果
# print(feature_X)
# print(feature_X.shape)
# print(scaler.fit_transform(X_data))  # 使用fit_transform(data)一步达成结果
#======================================================================
label_y = data[:,6]#标签作为数组的维度
# print(label_y)
print("删除无用数据并经过标准化后数据集中包含的数据总数：",len(label_y))#26156个数据
#======================================================================
#数据集划分为80%训练集,20%测试集,
X_train, X_test, y_train, y_test = train_test_split(feature_X, label_y,test_size=0.2,random_state=5)   #数据集——>训练验证集+测试集
#训练集,交叉验证集，测试集中的数据个数
print("80%的训练集数据:",len(y_train))#20924个训练数据
print("20%的测试集数据:",len(y_test))#5232个测试数据

from collections import Counter
# 查看所生成的样本类别分布，属于类别不平衡数据
print(Counter(y_train))

#=========利用imblearn库处理训练验证集上样本不均衡的问题，测试集依然保持原来的分布情况==========
# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
X_train, y_train = smo.fit_sample(X_train, y_train)
print("SMOT平衡后的数据总数",len(list(y_train)))

# #==========将处理后的训练数据，测试数据分开存储到CSV文件便于使用==============
# data_train = pd.DataFrame(np.hstack((X_train,y_train.reshape(-1,1))))#将上述的特征数据和处理好的分类标签数据合并到一起,并转化为DataFrame数据
# # print(data_train.head())
# data_train.to_csv('data_train.csv')
#
# data_test = pd.DataFrame(np.hstack((X_test,y_test.reshape(-1,1))))#将上述的特征数据和处理好的分类标签数据合并到一起,并转化为DataFrame数据
# # print(data_test.head())
# data_test.to_csv('data_test.csv')