# #================================导入机器学习过程所要用的库=====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV#导入网格搜索
import joblib
from sklearn.model_selection import cross_val_score#交叉验证评分
from sklearn.metrics import confusion_matrix

#==============导入已获得的训练测试文件================

data_train = pd.read_csv('data_train.csv',index_col=0).values
# print(data_train.shape)
X_train = data_train[:,:6]
# print(X_train.shape)
y_train = data_train[:,6]
# print(y_train.shape)

data_test = pd.read_csv('data_test.csv',index_col=0).values
# print(data_test.shape)
X_test = data_test[:,:6]
# print(X_test.shape)
y_test = data_test[:,6]
# print(y_test.shape)

# #===================用学习曲线来对决策树的数目进行调参==================
# #注range,np.arange第三个数据是所要生成数据的步长(start,finish)取不到finish
# #np.linspace第三个数据是所要生成数据的个数(start,fnish)取的到finish
# #最佳超参数3
# score_curve = []
# for i in range(1,10,1):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     score = cross_val_score(knn,X_train,y_train,cv=10).mean()
#     score_curve.append(score)
# plt.figure()
# plt.xticks(np.arange(1,10,1))
# plt.plot(range(1,10,1),score_curve)
# plt.title("Learning curve of KNN")
# plt.show()

# #========================设置优化参数进行网格搜索获取最优超参数并进行模型训练===============================
# from sklearn.model_selection import GridSearchCV#导入网格搜索
# from sklearn.neighbors import KNeighborsClassifier
# #利用sklearn中的库创建一个knn类
# #设置KNN所需优化的相关参数，并将这些参数置于字典中
# #这里只优化参数临近样本点数K
# knn = KNeighborsClassifier()
# k_range = range(1,5)
# #下面是构建parameter grid，其结构是key为参数名称，value是待搜索的数值列表的一个字典结构
# #定义优化参数字典，字典中的key值必须是分类算法的函数的参数名
# param_grid = {'n_neighbors':k_range}
# #打印待优化的参数
# #print(param_grid)
# #这里GridSearchCV的参数形式和cross_val_score的形式差不多，其中param_grid是parameter grid所对应的参数
# #GridSearchCV中的n_jobs设置为-1时，可以实现并行计算（如果你的电脑支持的情况下）
# #针对每个参数对进行了10次交叉验证。scoring='accuracy'使用准确率为结果的度量指
# #网格法搜寻最优超参数也涉及了交叉验证的思想(CV=10交叉验证集的个数)
# grid = GridSearchCV(estimator = knn, param_grid = param_grid, cv=10, scoring='accuracy')
# #用网格法在这组训练数据上寻找最优的参数
# grid.fit(X_train, y_train)
# print('网格搜索-最佳参数：',grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
# # print('网格搜索-最佳模型：',grid.best_estimator_)  # 获取最佳度量时的分类器模型
#
# # 使用获取的最佳参数生成模型，预测数据
# # 取出刚才在训练数据上得到的最佳参数进行建模
# knn = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
# knn.fit(X_train,y_train)  # 训练模型

# #========================保存训练好的模型，以及导入训练好的模型============================
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train,y_train)  # 训练模型
# # import joblib
# #保存训练好的模型，注:save文件夹要预先建立，否则会报错
# joblib.dump(knn, 'module_save/knn.pkl')
# 读取训练好的模型
knn2 = joblib.load('module_save/knn.pkl')
#==================================模型评估===================================
#训练验证集包括训练集和交叉验证集
from sklearn.model_selection import cross_val_score#交叉验证评分
# print(knn2.predict(X_test))
# print(y_test)
print("KNN最佳超参数n_neighbours:" + str(knn2.get_params()['n_neighbors']))
#===============================================================================
#分类模型评分标准不要用R-2方法
# print('网格搜索<有交叉验证>获得的最好估计器,在训练集上没做交叉验证的得分:',knn2.score(X_train,y_train))
# scores = cross_val_score(knn2, X_train, y_train, cv=10)   #在训练集和验证集上进行交叉验证
# print('网格搜索<有交叉验证>获得的最好估计器,在训练验证集上做交叉验证的平均得分:',np.mean(scores))   #交叉验证的平均accuracy
# print('网格搜索<有交叉验证>获得的最好估计器,在测试集上的得分:',knn2.score(X_test,y_test))#泛化能力


#==================================================================================
#========================可视化相关操作===============================
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#求混淆矩阵并将其可视化
#只针对于分类问题
y_predict = knn2.predict(X_test)
# print(y_predict)
print(list(y_test).count(0))#真实的MDI数据为808个
print(list(y_test).count(1))#真实的TF数据为4424个
print(list(y_predict).count(0))#预测的数据中841个MDI数据
print(list(y_predict).count(1))#预测的数据中4391个TF数据

#分类模型的评估标准混淆矩阵，准确率
confusion_mat=confusion_matrix(y_test, y_predict)
print(confusion_mat)
true_predict=confusion_mat[0,0]+confusion_mat[1,1]
accuracy = true_predict / len(list(y_predict))
print(accuracy)
print(round(accuracy,5))#0.99331

#定义混淆矩阵可视化函数
def plot_confusion_matrix(confusion_mat,classes):

    # 作图阶段
    # fig = plt.figure()
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    # ax = fig.add_subplot(111)
    plt.figure()
    # 定义横纵坐标的刻度以及对应标签
    # ax.set_yticks(range(len(classes)))
    # ax.set_yticklabels(classes)
    plt.yticks(range(len(classes)),classes)
    # ax.set_xticks(range(len(classes)))
    # ax.set_xticklabels(classes)
    plt.xticks(range(len(classes)), classes)
    plt.title('Confusion Matrix of KNN ')
    plt.ylabel('Predicted Label')
    plt.xlabel('True Label')
    #===========混淆矩阵可视化时要先设置参数在进行绘制===============
    plt.imshow(confusion_mat, cmap=plt.cm.get_cmap('Greys'))
    plt.colorbar()  # 附注上颜色对应的数值标度，要在绘制完图之后进行
    #=============================================================
    for first_index in range(len(confusion_mat)):
        for second_index in range(len(confusion_mat[first_index])):
            plt.text(first_index, second_index, confusion_mat[first_index][second_index], va='center', ha='center', color='red', fontsize=20)



plot_confusion_matrix(confusion_mat,['MDI','TF'])
plt.show()

