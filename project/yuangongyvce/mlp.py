# -*- coding: utf-8 -*-
# @Time: 2020/6/19 10:20
# @Author: wangshengkang
import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
import pandas as pd

data = pd.read_csv('train.csv')  # 读取训练集
print(data.head(5))

data = pd.DataFrame(data)  # 将数据格式变为DataFrame
# print(data.columns)
# predictor_cols=['Label']
train_y = data[['Label']]  # 训练集的标签
print(train_y.head(5))

# predictor_cols = ['Age', 'DistanceFromHome', 'Education', 'EmployeeNumber', 'EnvironmentSatisfaction',
#                  'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
#                  'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
#                  'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
#                  'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
# 要训练的真正的数据
# train_x = data[predictor_cols]
# train_x=data


# predictor_cols_onehot=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']
# train_onehot=data[predictor_cols_onehot]
# onehot=keras.utils.to_categorical(train_onehot,num_classes=num)
data = data.drop(['ID', 'Label'], 1)  # 将ID，标签列舍弃掉
print(data.head(5))
# 将训练集部分特征进行one-hot编码
train_x = pd.get_dummies(data, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
                                        'MaritalStatus', 'Over18', 'OverTime'])
print(train_x.head(5))

print(train_x.shape[1])
# 创建MLP模型
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(train_x.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 损失函数，评估标准，优化器
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

# 进行训练，20%的数据划出来作为验证集
model.fit(train_x, train_y, batch_size=100, epochs=30, verbose=2, validation_split=0.2)

test_data = pd.read_csv('test_noLabel.csv')  # 读取测试集
test_data = pd.DataFrame(test_data)  # 将测试集变为DataFrame形式
test_id = test_data[['ID']]  # 获取ID列
# print(test_id)
test_data = test_data.drop('ID', 1)  # 讲测试集的ID列删除
# 将测试集部分特征进行one-hot编码
test_x = pd.get_dummies(test_data,
                        columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
                                 'Over18', 'OverTime'])
print(test_x.head(5))

# 模型预测结果
test_y = model.predict(test_x)
# 将预测结果处理为DataFrame形式
test_y = pd.DataFrame(test_y)

# test_y['ID']=test_id[['ID']]
# print(test_y.head(5))

# 讲sigmoid的预测结果四舍五入，作为标签
test_y = np.round(test_y).astype(np.int)
print(test_y.head(5))
# 给标签加上ID列
test_y.insert(0, 'ID', test_id)

# test_id=test_id.narrdy
# test_y.reindex(columns=list('ID0'))
# pred=pd.concat(test_id,test_y)

# 将预测结果输出到csv文件中
test_y.to_csv('mlppred.csv', index=None, header=['ID', 'Label'])
