# -*- coding: utf-8 -*-
# @Time: 2020/6/15 11:12
# @Author: wangshengkang

import torchvision
from torchsummary import summary
# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理-----------------------------------------




vggmodel=torchvision.models.vgg16(pretrained=True,progress=True)
print(vggmodel)
summary(vggmodel, (3, 640, 640))

