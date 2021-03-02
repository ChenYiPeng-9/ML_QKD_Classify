#==================加入特征压缩(标准化)后的数据降维可视化=================
# import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#==================导入数据=======================
my_data = pd.read_csv('data_finsh.csv',index_col=0)
# print(my_data.head())
# print(my_data.info)
data = my_data.values
# print(data)
# print(data.shape)
X_data = data[:,:6]
# print(X_data)

#======对数据标准化======
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # 实例化
scaler.fit(X_data)  # fit，本质是生成均值和方差
X_data_std = scaler.transform(X_data)  # 通过接口导出结果
# print(X_data_std)
# print(scaler.fit_transform(data))  # 使用fit_transform(data)一步达成结果

#=======================
y_data = data[:,6]#标签作为数组的维度
# print(y_data)
# print("标签的个数为:",len(list(y_data)))#总共的个数是22116
# print("标签为0的数据是(MDI)：",list(y_data).count(0))#MDI的数据量4040个
# print("标签为1的数据是(TF)：",list(y_data).count(1))#TF的数据量是22116个
# from collections import Counter
# print(Counter(y_data))
# print(pd.DataFrame(X_data))#特征作为特征矩阵的维度

#=========利用imblearn库处理样本不均衡的问题==========
# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
X_data_std_smo, y_data_smo = smo.fit_sample(X_data_std, y_data)

# print("标签的个数为:",len(list(y_data_smo)))#总共的个数是44232
# print("标签为0的数据是(MDI)：",list(y_data_smo).count(0))#MDI的数据量22116个
# print("标签为1的数据是(TF)：",list(y_data_smo).count(1))#TF的数据量是22116个
# from collections import Counter
# print(Counter(y_data_std_smo))

#=================PCA建模并降维====================
pca = PCA(n_components=2)#实例化，最终生成两个特征
pca = pca.fit(X_data_std_smo)#拟合模型
new_X_data = pca.transform(X_data_std_smo)
# print(new_X_data)
# new_X_data = PCA(2).fit_transform(X_data)#上面三步可以一步到位的来写

#================对降维后的数据进行可视化================
# colors = ['red','black']
# names = ['KR1','KR4']
# for i in [0,1]:
#     plt.scatter(new_X_data[y_data==i,0],new_X_data[y_data==i,1]
#                 ,alpha=.7
#                 ,c=colors[i]
#                 ,label=names[i]
#                 )

# plt.figure()
# plt.scatter(new_X_data[y_data_smo==0,0],new_X_data[y_data_smo==0,1],s=0.1,c="red",label = "MDI",)
# plt.legend()
# plt.title("PCA of MDI ")
# plt.xlabel('feature one')
# plt.ylabel('feature two')
#
# plt.figure()
# plt.scatter(new_X_data[y_data_smo==1,0],new_X_data[y_data_smo==1,1],s=0.1,c="blue",label = "TF",)
# plt.legend()
# plt.title("PCA of TF")
# plt.xlabel('feature one')
# plt.ylabel('feature two')
# plt.show()


plt.figure()
plt.scatter(new_X_data[y_data_smo==0,0],new_X_data[y_data_smo==0,1],s=0.1,c="red",label = "MDI",)
plt.scatter(new_X_data[y_data_smo==1,0],new_X_data[y_data_smo==1,1],s=0.1,c="blue",alpha=0.6,label = "TF",)
plt.legend()
plt.title("PCA of MDI & TF")
plt.xlabel('feature one')
plt.ylabel('feature two')
plt.show()

