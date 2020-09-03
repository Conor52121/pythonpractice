# -*- coding: utf-8 -*-
# @Time: 2020/8/21 20:45
# @Author: wangshengkang
# @Software: PyCharm
import re

pattern = re.compile(r'([-\d]+)_c([-\d]+)')#创建一个正则匹配规则
result=pattern.search('-122_c1234').groups()
print(result)