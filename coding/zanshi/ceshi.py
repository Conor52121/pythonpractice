# -*- coding: utf-8 -*-
# @Time: 2020/6/11 17:11
# @Author: wangshengkang
import numpy as np
n=10
y = np.zeros([2 * n, 2])  # 构造判别器标签，one-hot编码
y[:n, 1] = 1  # 真实图像标签[1 0]
y[n:, 0] = 1  # 生成图像标签[0 1]
print(1)