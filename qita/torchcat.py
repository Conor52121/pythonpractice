# -*- coding: utf-8 -*-
# @Time: 2020/8/30 20:46
# @Author: wangshengkang
# @Software: PyCharm
import torch
import torch.nn
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

# x1=torch.cat((x, x, x), 0)
# x2=torch.cat((x, x, x), 1)
# print(x1.shape)
# print(x2.shape)

x = torch.randn(128,2048,8,4)
print('x1',x.shape)
x = F.avg_pool2d(x, x.size()[2:])  # 这是论文中所说的Pooling-5层 2048*1*1
print('x2', x.shape)

x2 = torch.randn(128,2048,4)
print('x3',x2.shape)
x2 = F.avg_pool1d(x2, x2.size()[2:])  # 这是论文中所说的Pooling-5层 2048*1*1
print('x4', x2.shape)

x3 = torch.randn(128,2048,4)
print('x5',x3.shape)
avgpool = torch.nn.AvgPool1d(4)  # 这是论文中所说的Pooling-5层 2048*1*1
x3=avgpool(x3)
print('x6', x3.shape)