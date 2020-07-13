# -*- coding: utf-8 -*-
# @Time: 2020/6/28 7:43
# @Author: wangshengkang
#这个版本是我手写的，目前还不能运行
# -------------------------------------------代码布局----------------------------------------------------
# 1导入相关包
# 2读取数据
# 3建立模型
# -------------------------------------------代码布局----------------------------------------------------
# -------------------------------------------1引用相关包----------------------------------------------------
import numpy as np
import torch.nn as nn
import argparse
import torch
import torchvision
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pickle
import os

# -------------------------------------------1引用相关包----------------------------------------------------
# -------------------------------------------2读取数据-------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = '5'  # 选择gpu
# python的pickle模块实现了基本的数据序列和反序列化。通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存
# 到文件中去，永久存储；通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。
# 从file中读取一个字符串，并将它重构为原来的python对象。file:类文件对象，有read()和readline()接口。
data_batch_1 = pickle.load(open('cifar-10-batches-py/data_batch_1', 'rb'), encoding='bytes')
data_batch_2 = pickle.load(open('cifar-10-batches-py/data_batch_2', 'rb'), encoding='bytes')
data_batch_3 = pickle.load(open('cifar-10-batches-py/data_batch_3', 'rb'), encoding='bytes')
data_batch_4 = pickle.load(open('cifar-10-batches-py/data_batch_4', 'rb'), encoding='bytes')
data_batch_5 = pickle.load(open('cifar-10-batches-py/data_batch_5', 'rb'), encoding='bytes')

train_X_1 = data_batch_1[b'data']
train_X_1 = train_X_1.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_1 = data_batch_1[b'labels']

train_X_2 = data_batch_2[b'data']
train_X_2 = train_X_2.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_2 = data_batch_2[b'labels']

train_X_3 = data_batch_3[b'data']
train_X_3 = train_X_3.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_3 = data_batch_3[b'labels']

train_X_4 = data_batch_4[b'data']
train_X_4 = train_X_4.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_4 = data_batch_4[b'labels']

train_X_5 = data_batch_5[b'data']
train_X_5 = train_X_5.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_5 = data_batch_5[b'labels']

train_X = np.row_stack((train_X_1, train_X_2))
train_X = np.row_stack((train_X, train_X_3))
train_X = np.row_stack((train_X, train_X_4))
train_X = np.row_stack((train_X, train_X_5))

train_Y = np.row_stack((train_Y_1, train_Y_2))
train_Y = np.row_stack((train_Y, train_Y_3))
train_Y = np.row_stack((train_Y, train_Y_4))
train_Y = np.row_stack((train_Y, train_Y_5))
train_Y = train_Y.reshape(50000, 1).transpose(0, 1).astype('int32')
#train_Y = keras.utils.to_categorical(train_Y)

test_batch = pickle.load(open('cifar-10-batches-py/test_batch', 'rb'), encoding='bytes')
test_X = test_batch[b'data']
test_X = test_X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
test_Y = test_batch[b'labels']
#test_Y = keras.utils.to_categorical(test_Y)

train_X /= 255
test_X /= 255


# -------------------------------------------2读取数据-------------------------------------------------------
# -------------------------------------------3建立模型-------------------------------------------------------
# 稠密层函数
class DenseLayer(nn.Module):
    def __init__(self,nb_filter,bn_size=4,drop_rate=0.2):
        super(DenseLayer,self).__init__()
        self.net1=nn.Sequential(
            nn.BatchNorm2d(nb_filter),
            nn.LeakyReLU(),
            nn.Conv2d(nb_filter,nb_filter*bn_size,(1,1),padding=None),

            nn.BatchNorm2d(nb_filter*bn_size),
            nn.LeakyReLU(),
            nn.Conv2d(nb_filter*bn_size,nb_filter,(3,3),padding=1)
        )
        self.drop_rate=drop_rate
    def forward(self,x):
        x_new=self.net1(x)
        if self.drop_rate: x_new=nn.Dropout(self.drop_rate)(x)
        return torch.cat([x,x_new],1)

class DenseBlock(nn.Module):
    def __init__(self,nb_filter,nb_layers,growth_rate,drop_rate=0.2):
        super(DenseBlock,self).__init__()
        for ii in range(nb_layers):
            net2=DenseLayer(nb_filter=growth_rate,drop_rate=drop_rate)
            self.add_module('denselayer%d'%(ii+1),net2)

class TransitionLayer(nn.Module):
    def __init__(self,nb_filter):
        super(TransitionLayer,self).__init__()
        net3=nn.Sequential(
            nn.BatchNorm2d(nb_filter),
            nn.LeakyReLU(),
            nn.Conv2d(nb_filter,nb_filter,(1,1),padding=None),
            nn.AvgPool2d((2,2))
        )

class DenseNet(nn.Module):
    def __init__(self,nb_filter,growth_rate,):
        super(DenseNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(nb_filter,growth_rate*2,(3,3),padding=1),
            nn.BatchNorm2d(nb_filter),
            nn.LeakyReLU()
        )
        self.DenseBlock=DenseBlock()
        self.TransitionLayer=TransitionLayer()
        self.fenlei=nn.Sequential(
            nn.BatchNorm2d(),
            nn.AvgPool2d(),
            nn.Linear(),
            nn.Softmax()
        )
    def forward(self,x):
        x=self.features(x)
        x=self.DenseBlock(x)
        x=self.TransitionLayer(x)
        x=self.DenseBlock(x)
        x=self.TransitionLayer(x)
        x=self.DenseBlock(x)
        x=self.fenlei(x)

        return x

model=DenseNet()