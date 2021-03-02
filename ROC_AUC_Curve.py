# #================================导入机器学习过程所要用的库=====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV#导入网格搜索
import joblib
from sklearn.model_selection import cross_val_score#交叉验证评分
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

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


#=======================================绘制ROC曲线===================================================
#================================上面这些只是单纯的得到测试集的数据=====================================
#========================接下来利用之前训练好的模型去获得测试集的得分====================================
dtc3 = joblib.load('module_save/dtc.pkl')
knn3 = joblib.load('module_save/knn.pkl')
rfc3 = joblib.load('module_save/rfc.pkl')
svc3 = joblib.load('module_save/svc.pkl')

###通过两种方式计算出模型预测数据为正例的分数(概率)，用在roc_curve()函数中
y_score_dtc = dtc3.predict_proba(X_test)
y_score_knn = knn3.predict_proba(X_test)
y_score_rfc = rfc3.predict_proba(X_test)
y_score_svc = svc3.decision_function(X_test)

# print(y_score_dtc[:50])
# print(y_score_rfc[:50])
#===========================================================================
# 利用roc_curve函数计算出绘制ROC曲线的真阳率tpr，假阳率fpr，以及相应的阈值
# X_test中的每一个数据都有一个阈值，通过这个阈值对数据集整体进行处理得到一组tpr，fpr
# X_test中若有n个数据则有n个阈值，则有n组tpr和fpr
# 以tpr为纵轴，以fpr为横轴，对上述获得数据可视化得到ROC曲线，ROC曲线下面的面积即auc
# ROC曲线越靠近左上角，对应面积auc的数值越大则该模型分类器的效果越好

#===================================================================================================================
# roc_curve(y_true, y_score, pos_label=None):
# y_true:实际的样本标签值，这里只能用来处理二分类问题，如果有多个标签要使用pos_label 指定某个标签为正例，其他的为反例）
# y_score:目标分数，被分类器识别成正例！！！的分数（常使用在method="decision_function"(SVM模型中)、method="proba_predict"）
# pos_label:指定某个标签为正例

#================计算绘制ROC曲线以及auc数值所需要的tpr和fpr======================
dtc_fpr, dtc_tpr, dtc_threshold = roc_curve(y_test, y_score_dtc[:,1],pos_label=1)  ###计算真阳率tpr和假阳率fpr
dtc_auc = auc(dtc_fpr, dtc_tpr)  ###计算auc的值

knn_fpr, knn_tpr, knn_threshold = roc_curve(y_test, y_score_knn[:,1],pos_label=1)  ###计算真阳率tpr和假阳率fpr
knn_auc = auc(knn_fpr, knn_tpr)  ###计算auc的值

rfc_fpr, rfc_tpr, rfc_threshold = roc_curve(y_test, y_score_rfc[:,1],pos_label=1)  ###计算真阳率tpr和假阳率fpr
rfc_auc = auc(rfc_fpr, rfc_tpr)  ###计算auc的值

svc_fpr, svc_tpr, svc_threshold = roc_curve(y_test, y_score_svc,pos_label=1)  ###计算真阳率tpr和假阳率fpr
svc_auc = auc(svc_fpr, svc_tpr)  ###计算auc的值


#================================ROC数据可视化=====================================
plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.plot(dtc_fpr,dtc_tpr,color='r', marker='o',markersize=3, linestyle='--', linewidth=0.5, label = "dtc_auc:" + str(dtc_auc))
plt.plot(rfc_fpr,rfc_tpr,color='y', marker='.',markersize=1, linestyle='--', linewidth=0.5, label = "rfc_auc:" + str(rfc_auc))
plt.plot(knn_fpr,knn_tpr,color='k', marker='.',markersize=1, linestyle='--', linewidth=0.5, label = "knn_auc:" + str(knn_auc))
plt.plot(svc_fpr,svc_tpr,color='b', marker='.',markersize=1,linestyle='--', linewidth=0.5, label = "svc_auc:" + str(svc_auc))

plt.legend()
plt.show()
