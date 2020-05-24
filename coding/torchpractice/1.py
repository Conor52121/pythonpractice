# -*- coding: utf-8 -*-
# @Time: 2020/5/20 10:43
# @Author: wangshengkang
# @Version: 1.0
# @FileName: 1.0.py
# @Software: PyCharm
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time

# 数据放到本地路径
path = 'boston_housing.npz'
f = np.load(path)
# 404个训练，102个测试
# 训练数据
x_train=f['x'][:404]  # 下标0到下标403
y_train=f['y'][:404]
# 测试数据
x_valid=f['x'][404:]  # 下标404到下标505
y_valid=f['y'][404:]
f.close()

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)

#  -------------------------- 3、数据归一化 -------------------------------
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

batch_size = 200
#train_set = ImgDataset(train_x, train_y, train_transform)
#val_set = ImgDataset(val_x, val_y, test_transform)
#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # 进行shuffle随机
#val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

print(x_train_pd.shape[1])
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(13, 10),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(10,15),
            nn.ReLU(),

            nn.Linear(15,1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.fc(x)
        return out

model = Classifier()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 200  # 迭代30次

for epoch in range(num_epoch):
    train_loss = 0.0

    model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(x_train):
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0])  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1])  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        train_loss += batch_loss.item()

        print(train_loss)

# test_set = ImgDataset(test_x, transform=test_transform)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#
# model.eval()
# prediction = []
# with torch.no_grad():
#     for i, data in enumerate(test_loader):
#         test_pred = model_best(data.cuda())
#         test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
#         for y in test_label:
#             prediction.append(y)