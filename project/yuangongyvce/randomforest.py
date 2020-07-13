# -*- coding: utf-8 -*-
# @Time: 2020/6/20 16:03
# @Author: wangshengkang

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')
#print(data.head(5))

data = pd.DataFrame(data)

train_y = data[['Label']]
#print(train_y.head(5))

data=data.drop(['ID','Label'],1)
#print(data.head(5))
train_x = pd.get_dummies(data,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime'])
#print(train_x.head(5))

#print(train_x.shape[1])
train_x,val_x,train_y,val_y=train_test_split(train_x,train_y,test_size=0.2,random_state=1)
rf=RandomForestClassifier()
rf.fit(train_x,train_y)
print('Random Forset Accuracy:{:.5f}'.format(accuracy_score(val_y,rf.predict(val_x))))

test_data=pd.read_csv('test_noLabel.csv')
test_data=pd.DataFrame(test_data)
test_id=test_data[['ID']]
#print(test_id)
test_data=test_data.drop('ID',1)
test_x = pd.get_dummies(test_data,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime'])
#print(test_x.head(5))

test_y=rf.predict(test_x)
test_y=pd.DataFrame(test_y)


test_y=np.round(test_y).astype(np.int)
#print(test_y.head(5))
test_y.insert(0,'ID',test_id)

test_y.to_csv('randomforestpred.csv',index=None, header=['ID','Label'])
