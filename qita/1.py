# -*- coding: utf-8 -*-
# @Time: 2020/7/19 19:30
# @Author: wangshengkang
# @Software: PyCharm
from PIL import Image
import torch
img=Image.open('jay.jpg')
print(img)
inv_idx = torch.arange(10 - 1, -1, -1).long()
print(inv_idx)