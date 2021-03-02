# #================================导入机器学习过程所要用的库=====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV#导入网格搜索
# from sklearn.externals import joblib
# 同样的joblib导入的方式不同训练出来的模型不能共用
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


#====================用网格法对核函数进行调参========================
# param_grid = {'kernel':('linear','poly','rbf','sigmoid')}
# svc = svm.SVC(gamma='auto',random_state=5)
# GS = GridSearchCV(svc,param_grid,cv=10)
# GS.fit(X_train,y_train)
# print(GS.best_params_)

#======================用获取的最佳超参数建立模型==================================
#这里的超参数都是默认的
# svc = svm.SVC(gamma='auto',random_state=5)
# svc.fit(X_train,y_train)  # 训练模型
# #
# #========================保存训练好的模型，以及导入训练好的模型============================
# # 保存训练好的模型，注:save文件夹要预先建立，否则会报错
# joblib.dump(svc, 'module_save/svc.pkl')
#读取训练好的模型
svc2 = joblib.load('module_save/svc.pkl')
#==================================模型评估===================================
#训练验证集包括训练集和交叉验证集
from sklearn.model_selection import cross_val_score#交叉验证评分
# print(svc2.predict(X_test))
# print(y_test)
print('最佳超参数kernel：' + str(svc2.get_params()['kernel']))

#分类模型评分标准不要用R-2方法
# print('网格搜索+学习曲线<有交叉验证>获得的最好估计器,在训练集上没做交叉验证的得分:',svc2.score(X_train,y_train))
# scores = cross_val_score(svc2, X_train, y_train, cv=10)   #在训练集和验证集上进行交叉验证
# print('网格搜索+学习曲线<有交叉验证>获得的最好估计器,在训练集上做交叉验证的平均得分:',np.mean(scores))   #交叉验证的平均accuracy
# print('网格搜索+学习曲线<有交叉验证>获得的最好估计器,在测试集上的得分:',svc2.score(X_test,y_test))#泛化能力

#========================可视化相关操作===============================
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#求混淆矩阵并将其可视化
#只针对于分类问题
y_predict = svc2.predict(X_test)
print(list(y_test).count(0))#真实的MDI数据为808个
print(list(y_test).count(1))#真实的TF数据为4424个
print(list(y_predict).count(0))#预测的数据中823个MDI数据
print(list(y_predict).count(1))#预测的数据中4409个TF数据

#分类模型的评估标准混淆矩阵，准确率
confusion_mat=confusion_matrix(y_test, y_predict)
print(confusion_mat)
true_predict=confusion_mat[0,0]+confusion_mat[1,1]
accuracy = true_predict / len(list(y_predict))
print(round(accuracy,5))#0.99713

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
    plt.title('Confusion Matrix of SVC')
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