# -*- coding: utf-8 -*-
# @Time: 2020/6/8 20:57
# @Author: wangshengkang
# -----------------------------------代码布局--------------------------------------------
# 1引入pytorch，numpy，matplotlib，IPython等包
# 2导入数据，数据预处理
# 3建立模型
# 4训练模型，预测结果
# 5结果以及损失函数可视化
# -----------------------------------代码布局--------------------------------------------
# ------------------------------------1引入包-----------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset,TensorDataset
# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理------------------------------------------
plt.switch_backend('agg')  # 服务器没有gui

path = 'mnist.npz'
f = np.load(path)
print(f.files)

X_train = f['x_train']
X_test = f['x_test']
f.close()

print(X_train.shape)  # (60000, 28, 28)
print(X_test.shape)  # (10000, 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32') / 255.  # 归一化
X_test = X_test.astype('float32') / 255.

noise_factor=0.5
X_train_noisy=X_train+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=X_train.shape)
X_test_noisy=X_test+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=X_test.shape)

X_train_noisy=np.clip(X_train_noisy,0.,1.)
X_test_noisy=np.clip(X_test_noisy,0.,1.)

X_train = torch.from_numpy((X_train))
X_test = torch.from_numpy((X_test))
X_train_noisy=torch.from_numpy(X_train_noisy)
X_test_noisy=torch.from_numpy(X_test_noisy)




# ------------------------------------2数据处理------------------------------------------
# ------------------------------------3建立模型------------------------------------------


class denoiseAE(nn.Module):
    def __init__(self):
        super(denoiseAE, self).__init__()
        self.denoise=nn.Sequential(
            nn.Conv2d(1,32,(3,3),padding=1),#28*28*32
            nn.ReLU(),
            nn.MaxPool2d((2,2)),#14*14*32
            nn.Conv2d(32,32,(3,3),padding=1),#14*14*32
            nn.ReLU(),
            nn.MaxPool2d((2,2)),#7*7*32

            nn.Conv2d(32,32,(3,3),padding=1),#7*7*32
            nn.ReLU(),
            nn.Upsample((14,14)),#14*14*32
            nn.Conv2d(32,32,(3,3),padding=1),#14*14*32
            nn.ReLU(),
            nn.Upsample((28,28)),#28*28*32
            nn.Conv2d(32,1,(3,3),padding=1),#28*28*1
        )

    def forward(self, x):
        out=denoiseAE(x)
        return out


model = denoiseAE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型，预测结果------------------------------------------
loss_total = []
epoch_total = []

epochs = 3
for epoch in range(epochs):
    pre = model(X_train_noisy)
    train_loss = loss(pre, X_train)
    loss_total.append(train_loss)
    epoch_total.append(epoch)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    print('epoch %3d, loss %10f' % (epoch, train_loss))
# ------------------------------------4训练模型，预测结果------------------------------------------
# ------------------------------------5可视化------------------------------------------

# RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
pre_test = model(X_test_noisy).detach().numpy()

n = 10
plt.figure(figsize=(20, 6))
for i in range(10):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(pre_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(fname='tu1.png')  # 保存图片
plt.show()

plt.plot(epoch_total, loss_total, label='loss')
plt.title('torch loss')  # 题目
plt.xlabel('Epoch')  # 横坐标
plt.ylabel('Loss')  # 纵坐标
plt.legend(['train'], loc='upper left')  # 图线示例
plt.savefig(fname='tu2.png')
plt.show()  # 画图

# ------------------------------------5可视化------------------------------------------

