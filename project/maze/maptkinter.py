# -*- coding: utf-8 -*-
# @Time: 2020/6/26 16:53
# @Author: wangshengkang
import random


from tkinter import *
from maptkinter import *
from drawmap import *
class TKDrawMap(DrawMap):
    def __init__(self, mmap):
        super(TKDrawMap, self).__init__(mmap, cell_width=10)
        master = Tk()#调用tkinter
        width, height = self.get_map_size()#获取尺寸大小
        self.w = Canvas(master, width=width, height=width)#调用Canvas
        self.w.pack()#显示
        self.draw_map()
        mainloop()#循环

    #按开始处红点按钮，开始画路线图
    def draw_start(self):
        r = self.cell_width // 3
        x, y = self.get_cell_center(0, 0)
        start = self.w.create_oval(x - r, y - r, x + r, y + r, fill="red")
        self.w.tag_bind(start, '<ButtonPress-1>', lambda e: self.draw_solution())

    #按结尾处红点按钮，更换地图
    def draw_end(self):
        r = self.cell_width // 3
        x, y = self.get_cell_center(self.mmap.max_x - 1, self.mmap.max_y - 1)
        end = self.w.create_oval(x - r, y - r, x + r, y + r, fill="red")
        self.w.tag_bind(end, '<ButtonPress-1>', lambda e: self.reset_map())

    #重画地图
    def reset_map(self):
        self.mmap.reset_map()
        self.w.delete('all')
        self.draw_map()

    def create_line(self, x1, y1, x2, y2, **kwargs):
        self.w.create_line(x1, y1, x2, y2, **kwargs)

    def create_solution_line(self, x1, y1, x2, y2):
        self.create_line(x1, y1, x2, y2, fill="red")