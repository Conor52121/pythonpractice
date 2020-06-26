# -*- coding: utf-8 -*-
# @Time: 2020/6/26 16:52
# @Author: wangshengkang
from __future__ import unicode_literals
import random
from tkinter import *
class Block(object):
    def __init__(self, mmap, x, y, direction=None):
        super(Block, self).__init__()
        #初始定义四个方向都没有堵塞
        self.walls = [True, True, True, True]  # top,right,bottom,left
        self.mmap = mmap
        if mmap:
            mmap.mmap[x][y] = self
        self.x, self.y = x, y
        #根据算法设置堵塞的墙
        if direction is not None:
            direction = (direction + 2) % 4
            self.walls[direction] = False

    def __unicode__(self):
        return "%s" % [1 if e else 0 for e in self.walls]

    def __str__(self):
        return unicode(self).encode('utf-8')

    #根据已有点的位置坐标，获取周围四个方向的坐标
    def get_next_block_pos(self, direction):
        x = self.x
        y = self.y
        if direction == 0:  # Top
            y -= 1
        elif direction == 1:  # Right
            x += 1
        if direction == 2:  # Bottom
            y += 1
        if direction == 3:  # Left
            x -= 1
        return x, y

    def get_next_block(self):
        directions = list(range(4))#四个方向
        random.shuffle(directions)#随机找一个方向
        for direction in directions:#每个方向遍历查找
            x, y = self.get_next_block_pos(direction)#下一个方向的坐标
            if x >= self.mmap.max_x or x < 0 or y >= self.mmap.max_y or y < 0:
                continue#可以往下走
            if self.mmap.mmap[x][y]:  # if walked
                continue#走了
            self.walls[direction] = False#此路不通
            return Block(self.mmap, x, y, direction)#记住堵塞的位置
        return None


class Map(object):
    def __init__(self):
        super(Map, self).__init__()

    # 重画地图
    def reset_map(self):
        self.gen_map(self.max_x, self.max_y)

    # 生成地图
    def gen_map(self, max_x=10, max_y=10):
        self.max_x, self.max_y = max_x, max_y
        self.mmap = [[None for j in range(self.max_y)] for i in range(self.max_x)]
        self.solution = []#定义路线的列表
        block_stack = [Block(self, 0, 0)]  # a unused block
        while block_stack:
            block = block_stack.pop()
            next_block = block.get_next_block()
            if next_block:
                block_stack.append(block)
                block_stack.append(next_block)
                #如果到了终点，开始循环加载路线图
                if next_block.x == self.max_x - 1 and next_block.y == self.max_y - 1:  # is end
                    for o in block_stack:
                        self.solution.append((o.x, o.y))#将正确的路线记录下来
    # python2用的
    def __unicode__(self):
        out = ""
        for y in range(self.max_y):
            for x in range(self.max_x):
                out += "%s" % self.mmap[x][y]
            out += "\n"
        return out

    # python3用的
    def __str__(self):
        return unicode(self).encode('utf-8')