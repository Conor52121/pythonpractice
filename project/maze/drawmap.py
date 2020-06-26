# -*- coding: utf-8 -*-
# @Time: 2020/6/26 16:53
# @Author: wangshengkang
class DrawMap(object):
    def __init__(self, mmap, cell_width=10):
        super(DrawMap, self).__init__()
        self.mmap = mmap
        self.cell_width = cell_width

    #地图尺寸
    def get_map_size(self):
        # width, height
        return (self.mmap.max_x + 2) * self.cell_width, (self.mmap.max_y + 2) * self.cell_width
    #画地图的线
    def create_line(self, x1, y1, x2, y2, **kwarg):
        raise NotImplemented()
    #行进路线
    def create_solution_line(self, x1, y1, x2, y2):
        self.create_line(x1, y1, x2, y2)

    def draw_start(self):
        raise NotImplemented()

    def draw_end(self):
        raise NotImplemented()

    def get_cell_center(self, x, y):
        w = self.cell_width
        return (x + 1) * w + w // 2, (y + 1) * w + w // 2

    def draw_solution(self):
        pre = (0, 0)#起点位置
        for o in self.mmap.solution:
            p1 = self.get_cell_center(*pre)#起始点位置
            p2 = self.get_cell_center(*o)#下一个点的位置
            self.create_solution_line(p1[0], p1[1], p2[0], p2[1])
            pre = o#起始点为上一个点

    def draw_cell(self, block):
        width = self.cell_width
        x = block.x + 1
        y = block.y + 1
        walls = block.walls
        if walls[0]:
            self.create_line(x * width, y * width, (x + 1) * width + 1, y * width)
        if walls[1]:
            self.create_line((x + 1) * width, y * width, (x + 1) * width, (y + 1) * width + 1)
        if walls[2]:
            self.create_line(x * width, (y + 1) * width, (x + 1) * width + 1, (y + 1) * width)
        if walls[3]:
            self.create_line(x * width, y * width, x * width, (y + 1) * width + 1)
    #画图
    def draw_map(self):
        for y in range(self.mmap.max_y):
            for x in range(self.mmap.max_x):
                self.draw_cell(self.mmap.mmap[x][y])
        self.draw_start()
        self.draw_end()