# -*- coding: utf-8 -*-
"""
 @Time    : 2022/4/20 11:47
 @Author  : meehom
 @Email   : meehomliao@163.com
 @File    : LOF.py
 @Software: PyCharm
"""
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.neighbors import LocalOutlierFactor

data = np.load('kdd_data_10_percent_preprocessed.npy', allow_pickle=True)
print(data)
train_X,test_X, train_y, test_y = train_test_split(data[:20000, :-1],
                                                   data[:20000,-1],
                                                   test_size = 0.2,
                                                   random_state = 0)
train_y = train_y.astype(int)
test_y = test_y.astype(int)


clf = LocalOutlierFactor(n_neighbors=5)
train_predict = clf.fit_predict(train_X)
test_predict = clf.fit_predict(test_X)


train_predict = [1 if x == 1 else 0 for x in train_predict]


test_predict = [1 if x == 1 else 0 for x in test_predict]

# train
print("train accuracy is :", accuracy_score(train_predict, train_y))
print("train f1_score is : ", f1_score(train_predict, train_y))
print("train recall_score is : ", recall_score(train_predict, train_y))
print("train precision_score is : ", precision_score(train_predict, train_y))

print("------------------------------------------")

# test
print("test accuracy is :", accuracy_score(test_predict, test_y))
print("test f1_score is : ", f1_score(test_predict, test_y))
print("test recall_score is : ", recall_score(test_predict, test_y))
print("test precision_score is : ", precision_score(test_predict, test_y))

