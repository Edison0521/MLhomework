import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# 以 预测准确率=（预测正确样本数）/（总测试样本数）* 100% 对预测准确率进行计算，设定 ErrorTolerance = 5%
def accuracy(predict, true):
    sizeofall = len(true)
    sizeofright = 0
    for i in range(0, sizeofall):
        est = abs(predict[i] - true[i]) / true[i]
        if est < 0.05:
            sizeofright = sizeofright + 1
    return sizeofright / sizeofall
# 导入数据
data = pd.read_csv('2.csv',encoding='utf-8')
used_features = ['收盘价','最高价','最低价','开盘价','前收盘','涨跌额']
X = data[used_features]
y = data['涨跌幅']
k=0

# 用交叉验证的方式从数据集中取40%作为测试集，其他作为训练集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=0,
)

# 创建SVM模型
svc = SVC()
# 用训练集训练模型
svc.fit(X_train, y_train.astype('int'))
# 用训练得出的模型进行预测
diabetes_y_pred = svc.predict(X_test)
# 将预测准确率打印出来
predict = np.array(diabetes_y_pred)
true = np.array(y_test)
Ac = accuracy(predict, true)
print("Accuracy=", Ac * 100, '%')
# 用matplotlib画图
plt.figure()
plt.plot(range(len(diabetes_y_pred)), diabetes_y_pred, label="predict value")
plt.plot(range(len(diabetes_y_pred)), y_test, label="true value")
plt.show()
# 创建朴素贝叶斯
gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train.astype(int)).predict(X_train)
predict = y_pred
true = np.array(y_test)
Ac = accuracy(predict, true)
print("Accuracy=", Ac * 100, '%')
