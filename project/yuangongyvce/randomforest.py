# -*- coding: utf-8 -*-
# @Time: 2020/6/20 16:03
# @Author: wangshengkang
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
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('train.csv')
print(data.head(5))

data = pd.DataFrame(data)

train_y = data[['Label']]
print(train_y.head(5))

data=data.drop(['ID','Label'],1)
print(data.head(5))
train_x = pd.get_dummies(data,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime'])
print(train_x.head(5))

print(train_x.shape[1])

rf=RandomForestClassifier()
rf.fit(train_x,train_y)


test_data=pd.read_csv('test_noLabel.csv')
test_data=pd.DataFrame(test_data)
test_id=test_data[['ID']]
print(test_id)
test_data=test_data.drop('ID',1)
test_x = pd.get_dummies(test_data,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime'])
print(test_x.head(5))

test_y=rf.predict(test_x)
test_y=pd.DataFrame(test_y)


test_y=np.round(test_y).astype(np.int)
print(test_y.head(5))
test_y.insert(0,'ID',test_id)

test_y.to_csv('rfpred.csv',index=None, header=['ID','Label'])
