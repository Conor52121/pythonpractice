# -*- coding: utf-8 -*-
# @Time: 2020/6/15 18:32
# @Author: wangshengkang
# 改动版本

import tkinter as tk  # 使用Tkinter前需要先导入
import tkinter.messagebox
import pickle

window = tk.Tk()
window.title('王盛康机器学习期末大作业')
window.geometry('1200x600')  # 这里的乘是小x
tk.Label(window, text='机器学习期末大作业', font=('Arial', 16)).place(x=300,y=20)
tk.Label(window, text='姓名:王盛康', font=('Arial', 14)).place(x=300, y=170)
tk.Label(window, text='班级:电子与通信工程', font=('Arial', 14)).place(x=300, y=210)
tk.Label(window, text='学号:201932071', font=('Arial', 14)).place(x=300, y=250)

def nextpage():
    pass

btn_login = tk.Button(window, text='查看作业',command=nextpage)
btn_login.place(x=300, y=300)
window.mainloop()