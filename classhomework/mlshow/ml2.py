# -*- coding: utf-8 -*-
# @Time: 2020/6/15 18:28
# @Author: wangshengkang

from tkinter import *
# Python3.x 导入方法
# from tkinter import *
root = Tk()  # 创建窗口对象的背景色
root.title('王盛康机器学习期末大作业')
root.geometry('800x500') # 这里的乘号不是 * ，而是小写英文字母 x
# 创建两个列表
li = ['C', 'python', 'php', 'html', 'SQL', 'java']
movie = ['CSS', 'jQuery', 'Bootstrap']
listb = Listbox(root)  # 创建两个列表组件
listb2 = Listbox(root)
for item in li:  # 第一个小部件插入数据
    listb.insert(0, item)

for item in movie:  # 第二个小部件插入数据
    listb2.insert(0, item)

listb.pack()  # 将小部件放置到主窗口中
listb2.pack()
root.mainloop()  # 进入消息循环