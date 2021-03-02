import pandas as pd
import numpy as np

#导入csv文件，data.csv文件从matlab处理后获取的
data = pd.read_csv('data.csv')
# print(data.head())#打印读取文件的前五行数据，以此来初步探索数据集的情况
# print(data.info())#打印该数据集内数据的具体情况，可以知道有的特征数据缺失了

#删除两个码率都错的那一个数据
alist=[0.0]#所要查找的数据0
index = data['6'].isin(alist) & data['7'].isin(alist)#获取码率都出错的数据所在行的索引
data_need = data[~index]#索引取反，获取错误数据之外的数据
# print(data_need.head())#正确数据的前五行
# print(data_need.values.shape)#正确数据的形状(11800,8)
X_data = data_need.values[:,:6]#DataFrame转换为数组后获取特征数据
label_data = data_need.values[:,6:8]#DataFrame转换为数组后获取初步分类数据
y_data = label_data.argmax(axis=1).reshape(-1,1)#对初步的分类数据进行比较获取分类标签(标签按每一行最大值的索引得到（一列），
#两个协议的密钥率谁在前面谁就是标签0，谁在后面谁就是标签1
# print(y_data)
# 一维数组，为了和特征合并转化为二维数组）
data_finsh = pd.DataFrame(np.hstack((X_data,y_data)))#将上述的特征数据和处理好的分类标签数据合并到一起,并转化为DataFrame数据
# print(data_finsh)
data_finsh.to_csv('data_finsh.csv')
