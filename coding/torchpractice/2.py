# -*- coding: utf-8 -*-
# @Time: 2020/5/20 20:28
# @Author: wangshengkang
# @Version: 1.0
# @FileName: 2.py
# @Software: PyCharm
'''
代码布局
1.导入需要的包
2.读取数据
3.数据预处理
4.模型构建
5.训练模型
6.结果可视化
'''
#%%
# 导入需要的包文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
#%%
# 读取数据
path = "boston_housing.npz"  #数据的路径
data = np.load(path)#加载npz数据
print(data.files)#打印数据包含的表头
# 获得数据，并查看数据类型
y = data['y']
x = data['x']
print(y.shape)
print(x.shape)
#%%
# 划分数据集（训练集和验证集）
train_x = x[:404]
test_x = x[404:]
train_y = y[:404]
test_y = y [404:]
data.close()
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)
#%%
# 数据归一化
# 转成DataFrame格式方便数据处理(矩阵类型)
x_train_pd = pd.DataFrame(train_x)#（404*13）
y_train_pd = pd.DataFrame(train_y)#（404*1）
x_valid_pd = pd.DataFrame(test_x)#（102*13）
y_valid_pd = pd.DataFrame(test_y)#（102*1）
# 训练集归一化归一到0-1之间
min_max_scaler = MinMaxScaler()

min_max_scaler.fit(x_train_pd)
train_x = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
train_y = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
test_x = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
test_y = min_max_scaler.transform(y_valid_pd)
# 上述数据处理结果为numpy类型
# 转换为tensor类型
print(train_x)
print('aaaaaaaaaaaaaaaaaaaaa')
train_x = torch.FloatTensor(train_x)
train_y = torch.FloatTensor(train_x)
test_x = torch.FloatTensor(train_x)
test_y = torch.FloatTensor(train_x)
#%%
print(train_x)
# 模型构建
# 三层网络结果
# 输入，输出，隐藏层
model = torch.nn.Sequential(
#     输入特征为X的13维特征，输出设定为10
    torch.nn.Linear(13, out_features=8,bias=False),
#     通过一个relu激活函数
    torch.nn.ReLU(),
#     根据上一个输入10，定义为一个输出
    torch.nn.Linear(8, 1,bias=False),
#     再跟随一个relu激活函数
    torch.nn.ReLU(),
)
# 定义学习率
learning_rate = 1e-3
# 定义损失函数MSE
loss_fn = torch.nn.MSELoss(reduction='sum')
# 优化函数Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#%%
num_epochs = 300
for i in range(num_epochs):
    y_pred =  model(train_x)
    # Compute and print loss.
    loss = loss_fn(y_pred,train_y)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate*param.grad
    #model.zero_grad()
    plt.plot(i,loss.item())
    plt.scatter(i,loss.item())
plt.show()


#%%
model()

#%%
