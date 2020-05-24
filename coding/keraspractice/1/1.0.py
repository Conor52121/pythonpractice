# ----------------------开发者信息-----------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/5/20 15:45
# @Author: wangshengkang
# @Version: 1.0
# @FileName: 1.0.py
# @Software: PyCharm
# ----------------------开发者信息-----------------------------------------
# ----------------------代码布局-------------------------------------
#1.引入keras，matplotlib，numpy，sklearn，pandas包
#2.导入数据
#3.数据归一化
#4.模型建立
#5.损失函数可视化
#6.保存模型预测结果
#---------------------------------------------------------------------
#---------------------------1引入相关包--------------------------------
from keras.preprocessing import sequence
from keras.models import Sequential
#from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#-----------------------------1引入相关包-----------------------------

#------------------------------2导入数据-----------------------------
path = 'boston_housing.npz'
f = np.load(path)

x_train=f['x'][:404]
y_train=f['y'][:404]

x_valid=f['x'][404:]
y_valid=f['y'][404:]
f.close()

x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))
print('------------------------')
print(y_train_pd.head(5))
#-------------------------------2导入数据--------------------------------

#-------------------------------3数据归一化------------------------------
# MinMaxScaler：归一到 [ 0，1 ]
# MaxAbsScaler：归一到 [ -1，1 ]
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)  # 计算最大值最小值，用来scale
x_train = min_max_scaler.transform(x_train_pd)  # scale
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)
min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)
#----------------------------3数据归一化-----------------------------

#-----------------------------4模型建立-------------------------------
print('aaaaaaaaaaaaaaaaaaa')
print(x_train_pd.shape[1])
model = Sequential()
model.add(Dense(units = 10,
                activation='relu',
                input_shape=(x_train_pd.shape[1],)
                )
          )

model.add(Dropout(0.2))

model.add(Dense(units=15,
                activation='relu',
                )
          )

model.add(Dense(units=1,
                activation='linear'
                )
          )

print(model.summary())

model.compile(loss='mse',
              optimizer='adam',
              )

history = model.fit(x_train,y_train,
                    epochs=200,
                    batch_size=200,
                    verbose=2,
                    validation_data = (x_valid, y_valid)
                    )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

from keras.utils import plot_model
from keras.models import load_model
model.save('model_MLP.h5')
plot_model(model, to_file='model_MLP.png', show_shapes=True)
model = load_model('model_MLP.h5')
y_new = model.predict(x_valid)
min_max_scaler.fit(y_valid_pd)
y_new =  min_max_scaler.inverse_transform(y_new)
y_new_pd = pd.DataFrame(y_new)
print(y_new_pd.head(5))