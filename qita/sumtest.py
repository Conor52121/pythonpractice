# -*- coding: utf-8 -*-
# @Time: 2020/6/29 10:59
# @Author: wangshengkang
# @Software: PyCharm
import torch
#测试max()作用
# a=torch.randn(5,7)
# print(a)
# _,c=torch.max(a,1)
# print(_)
# print(c)

#测试sum(),item()作用
b = torch.tensor([[1,2,3,4,5,],[6,7,8,9,10]])
print(b)
d=torch.tensor([[1,2,3,1,1,],[1,7,8,9,10]])
print(d)
correct=0
correct += (b == d).sum().item()
print('correct',correct)