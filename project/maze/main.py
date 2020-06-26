# -*- coding: utf-8 -*-
# @Time: 2020/6/26 16:53
# @Author: wangshengkang
from map import *
from drawmap import *
from tkinter import *
from maptkinter import *


def main():
    m = Map()  # 创建对象
    m.gen_map(20, 20)  # 调用方法，生成地图
    TKDrawMap(m)  # 画出路线图


if __name__ == '__main__':
    main()
