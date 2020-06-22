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

data = pd.read_csv('train.csv')
print(data.head(5))

data = pd.DataFrame(data)
#print(data.columns)
#predictor_cols=['Label']
train_y = data[['Label']]
print(train_y.head(5))

#predictor_cols = ['Age', 'DistanceFromHome', 'Education', 'EmployeeNumber', 'EnvironmentSatisfaction',
#                  'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
#                  'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
#                  'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
#                  'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
# 要训练的真正的数据
#train_x = data[predictor_cols]
#train_x=data


#predictor_cols_onehot=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']
#train_onehot=data[predictor_cols_onehot]
# onehot=keras.utils.to_categorical(train_onehot,num_classes=num)
data=data.drop(['ID','Label'],1)
print(data.head(5))
train_x = pd.get_dummies(data,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime'])
print(train_x.head(5))




print(train_x.shape[1])


model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(train_x.shape[1],)))
#model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

model.fit(train_x, train_y, batch_size=100, epochs=30, verbose=2,validation_split=0.2)


test_data=pd.read_csv('test_noLabel.csv')
test_data=pd.DataFrame(test_data)
test_id=test_data[['ID']]
print(test_id)
test_data=test_data.drop('ID',1)
test_x = pd.get_dummies(test_data,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime'])
print(test_x.head(5))

test_y=model.predict(test_x)
test_y=pd.DataFrame(test_y)

#test_y['ID']=test_id[['ID']]
#print(test_y.head(5))
test_y=np.round(test_y).astype(np.int)
print(test_y.head(5))
test_y.insert(0,'ID',test_id)
#test_id=test_id.narrdy
#test_y.reindex(columns=list('ID0'))
#pred=pd.concat(test_id,test_y)
test_y.to_csv('pred.csv',index=None, header=['ID','Label'])
