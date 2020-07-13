# -*- coding: utf-8 -*-
# @Time: 2020/6/30 12:45
# @Author: wangshengkang
# @Software: PyCharm
from tkinter import *

master = Tk()

var = StringVar(master)
var.set("one")  # initial value

option = OptionMenu(master, var, "one", "two", "three", "four")
option.pack()


#
# test stuff

def ok():
    print(
    "value is", var.get())
    master.quit()


button = Button(master, text="OK", command=ok)
button.pack()

mainloop()