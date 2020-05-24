import pandas as pd
import numpy as np
from pandas import DataFrame
import math
train=pd.read_csv("C:\\Project\\py\\airquality\\train.csv");
# pm=train[train[""]]
pm=train[train["testmaterial"]=="PM2.5"]
pm.drop(['date','station','testmaterial'],axis=1,inplace=True)
x=[]# 此时x,y为list
y=[]
for i in range(15):#冒号
    temx=pm.iloc[:,i:i+9]
    temx.columns=np.array(range(9))
    temy=pm.iloc[:,i+9]
    temy.columns=np.array(range(1))
    x.append(temx)
    y.append(temy)

x=pd.concat(x)#3600x9,3600=240x15,此时x为dataframe
y=pd.concat(y)#3600x1 ,y为serise
x=np.array(x,float)
y=np.array(y,float)
np.save("C:\\Project\\py\\airquality\\x.npy",x)
np.save("C:\\Project\\py\\airquality\\y.npy",y)

x=np.load("C:\\Project\\py\\airquality\\x.npy")
y=np.load("C:\\Project\\py\\airquality\\y.npy")
#adding baias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
#init
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 10000
s_grad=np.zeros(len(x[0]))
x_t=x.transpose()
# train
for i in range(repeat):
    tem=np.dot(x,w)
    loss=tem-y
    grad=np.dot(x_t,loss)
    s_grad+=grad**2
    ada=np.sqrt(s_grad)
    w=w-l_rate*grad/ada
np.save("C:\\Project\\py\\airquality\\model.npy",w)

model=np.load("C:\\Project\\py\\airquality\\model.npy")
test=pd.read_csv("C:\\Project\\py\\airquality\\test.csv")
t=test[test["testmaterial"]=="PM2.5"]
t.drop(["date","testmaterial"],axis=1,inplace=True)
t=np.array(t,float)
t=np.concatenate((np.ones((t.shape[0],1)),t), axis=1)
res=[]
res=np.dot(t,w)
res
