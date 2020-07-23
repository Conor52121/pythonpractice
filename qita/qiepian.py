# -*- coding: utf-8 -*-
# @Time: 2020/7/18 21:07
# @Author: wangshengkang
# @Software: PyCharm
import torch
from torch.autograd import Variable

part = {}
x = Variable(torch.FloatTensor(8, 2048, 6))
for i in range(6):
    part[i] = torch.squeeze(x[:,:, i])
print(part[0])
print(1)
print(1)
