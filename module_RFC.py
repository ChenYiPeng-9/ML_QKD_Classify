# #================================导入机器学习过程所要用的库=====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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


#===================用学习曲线来对决策树的数目进行调参==================
#注range,np.arange第三个数据是所要生成数据的步长(start,finish)取不到finish
#np.linspace第三个数据是所要生成数据的个数(start,fnish)取的到finish

# 大范围的学习曲线50-60
# score_curve = []
# for i in range(0,200,10):
#     rfc = RandomForestClassifier(n_estimators=i+1,random_state=5)#i+1!!!!
#     score = cross_val_score(rfc,X_train,y_train,cv=10).mean()
#     score_curve.append(score)
# plt.figure()
# plt.xticks(np.arange(0,201,10))
# plt.plot(range(0,200,10),score_curve)
# plt.show()
#小范围的学习曲线5
# score_curve = []
# for i in range(1,10):
#     rfc = RandomForestClassifier(n_estimators=i,random_state=5)#i!!!!!
#     score = cross_val_score(rfc,X_train,y_train,cv=10,).mean()
#     score_curve.append(score)
# plt.figure()
# plt.xticks(np.arange(1,10,1))
# plt.plot(range(1,10),score_curve)
# plt.show()

#====================用网格法结合学习曲线对树的深度进行调参========================
# param_grid = {'max_depth':np.arange(8,15,1)}#深度13
# rfc = RandomForestClassifier(n_estimators=5,random_state=5)
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(X_train,y_train)
# print(GS.best_params_)
# score_curve = []#8
# for i in range(1,10):
#     rfc = RandomForestClassifier(n_estimators=5,max_depth=i,random_state=0)
#     score = cross_val_score(rfc,X_train,y_train,cv=10).mean()
#     score_curve.append(score)
# plt.figure()
# plt.xticks(np.arange(1,10,1))
# plt.plot(range(1,10),score_curve)
# plt.show()

#======================用获取的最佳超参数建立模型==================================
# rfc = RandomForestClassifier(n_estimators=5,max_depth=8,random_state=5)
# rfc.fit(X_train,y_train)  # 训练模型
#
# # #========================保存训练好的模型，以及导入训练好的模型============================
# #保存训练好的模型，注:save文件夹要预先建立，否则会报错
# joblib.dump(rfc, 'module_save/rfc.pkl')
#读取训练好的模型
rfc2 = joblib.load('module_save/rfc.pkl')
#==================================模型评估===================================
#训练验证集包括训练集和交叉验证集
from sklearn.model_selection import cross_val_score#交叉验证评分
# print(rfc2.predict(X_test))
# print(y_test)
print('最佳超参数n_estimators：' + str(rfc2.get_params()['n_estimators']))
print('最佳超参数max_depth：'+ str(rfc2.get_params()['max_depth']))

#分类模型评分标准不要用R-2方法
# print('网格搜索+学习曲线<有交叉验证>获得的最好估计器,在训练集上没做交叉验证的得分:',rfc2.score(X_train,y_train))
# scores = cross_val_score(rfc2, X_train, y_train, cv=10)   #在训练集和验证集上进行交叉验证
# print('网格搜索+学习曲线<有交叉验证>获得的最好估计器,在训练集上做交叉验证的平均得分:',np.mean(scores))   #交叉验证的平均accuracy
# print('网格搜索+学习曲线<有交叉验证>获得的最好估计器,在测试集上的得分:',rfc2.score(X_test,y_test))#泛化能力

# #========================可视化相关操作===============================
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#求混淆矩阵并将其可视化
#只针对于分类问题
y_predict = rfc2.predict(X_test)
print(list(y_test).count(0))#真实的MDI数据为808个
print(list(y_test).count(1))#真实的TF数据为4424个
print(list(y_predict).count(0))#预测的数据中837个MDI数据
print(list(y_predict).count(1))#预测的数据中4395个TF数据

#分类模型的评估标准混淆矩阵，准确率
confusion_mat=confusion_matrix(y_test, y_predict)
print(confusion_mat)
true_predict=confusion_mat[0,0]+confusion_mat[1,1]
accuracy = true_predict / len(list(y_predict))
print(accuracy)
print(round(accuracy,5))#0.99446

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
    plt.title('Confusion Matrix of RandomForestClassifier ')
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