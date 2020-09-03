# -*- coding: utf-8 -*-
# @Time: 2020/7/19 19:30
# @Author: wangshengkang
# @Software: PyCharm
from PIL import Image
import torch
# img=Image.open('jay.jpg')
# print(img)
# print(img.size)
# #print(img.size(0))
# inv_idx = torch.arange(10 - 1, -1, -1).long()
# print(inv_idx)


filename='0006_c1s6_026921_00'
filename_split=filename.split('c')
print(filename_split)
camera_id = filename.split('c')[1][0]
print(camera_id)

# import os
# print((os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))