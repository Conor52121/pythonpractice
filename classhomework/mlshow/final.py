# -*- coding: utf-8 -*-
# @Time: 2020/6/15 19:01
# @Author: wangshengkang
#正式版本
import tkinter as tk


class basedesk():
    def __init__(self, master):
        self.root = master
        self.root.config()
        self.root.title('王盛康机器学习期末大作业')
        self.root.geometry('1200x600')

        initface(self.root)


class initface():
    def __init__(self, master):
        self.master = master
        # 基准界面initface
        self.initface = tk.Frame(self.master, )
        self.initface.pack()
        label1 = tk.Label(self.initface, text='机器学习期末大作业', font=('Arial', 16)).pack()
        label2 = tk.Label(self.initface, text='姓名:王盛康', font=('Arial', 14)).pack()
        label3 = tk.Label(self.initface, text='班级:电子与通信工程', font=('Arial', 14)).pack()
        label4 = tk.Label(self.initface, text='学号:201932071', font=('Arial', 14)).pack()
        btn = tk.Button(self.initface, text='下一页', command=self.change).pack()

    def change(self, ):
        self.initface.destroy()
        face_mulu(self.master)


class face_mulu():
    def __init__(self, master):
        self.master = master
        self.face1 = tk.Frame(self.master, )
        self.face1.pack()
        label1 = tk.Label(self.face1, text='目录', font=('Arial', 16)).pack()
        label2 = tk.Label(self.face1, text='摘要', font=('Arial', 16)).pack()
        label3 = tk.Label(self.face1, text='背景意义', font=('Arial', 16)).pack()
        label4 = tk.Label(self.face1, text='方法', font=('Arial', 16)).pack()
        label5 = tk.Label(self.face1, text='结果与分析', font=('Arial', 16)).pack()
        label6 = tk.Label(self.face1, text='展望', font=('Arial', 16)).pack()
        label7 = tk.Label(self.face1, text='收获', font=('Arial', 16)).pack()
        btn = tk.Button(self.face1, text='下一页', command=self.change).pack()
        btn_back = tk.Button(self.face1, text='上一页', command=self.back).pack()

    def change(self):
        self.face1.destroy()
        face_zhaiyao(self.master)

    def back(self):
        self.face1.destroy()
        initface(self.master)

class face_zhaiyao():
    def __init__(self, master):
        self.master = master
        self.face1 = tk.Frame(self.master, )
        self.face1.pack()
        label1 = tk.Label(self.face1, text='摘要', font=('Arial', 16)).pack()
        btn = tk.Button(self.face1, text='下一页', command=self.change).pack()
        btn_back = tk.Button(self.face1, text='上一页', command=self.back).pack()

    def change(self):
        self.face1.destroy()
        face_beijing(self.master)

    def back(self):
        self.face1.destroy()
        face_mulu(self.master)


class face_beijing():
    def __init__(self, master):
        self.master = master
        self.face1 = tk.Frame(self.master, )
        self.face1.pack()
        label1 = tk.Label(self.face1, text='背景意义', font=('Arial', 16)).pack()
        btn = tk.Button(self.face1, text='下一页', command=self.change).pack()
        btn_back = tk.Button(self.face1, text='上一页', command=self.back).pack()

    def change(self):
        self.face1.destroy()
        face_fangfa(self.master)

    def back(self):
        self.face1.destroy()
        face_zhaiyao(self.master)


class face_fangfa():
    def __init__(self, master):
        self.master = master
        self.face1 = tk.Frame(self.master, )
        self.face1.pack()
        label1 = tk.Label(self.face1, text='方法', font=('Arial', 16)).pack()
        btn = tk.Button(self.face1, text='下一页', command=self.change).pack()
        btn_back = tk.Button(self.face1, text='上一页', command=self.back).pack()

    def change(self):
        self.face1.destroy()
        face_jieguo(self.master)

    def back(self):
        self.face1.destroy()
        face_beijing(self.master)


class face_jieguo():
    def __init__(self, master):
        self.master = master
        self.face1 = tk.Frame(self.master, )
        self.face1.pack()
        label1 = tk.Label(self.face1, text='结果与分析', font=('Arial', 16)).pack()
        btn = tk.Button(self.face1, text='下一页', command=self.change).pack()
        btn_back = tk.Button(self.face1, text='上一页', command=self.back).pack()

    def change(self):
        self.face1.destroy()
        face_zhanwang(self.master)

    def back(self):
        self.face1.destroy()
        face_fangfa(self.master)


class face_zhanwang():
    def __init__(self, master):
        self.master = master
        self.face1 = tk.Frame(self.master, )
        self.face1.pack()
        label1 = tk.Label(self.face1, text='展望', font=('Arial', 16)).pack()
        btn = tk.Button(self.face1, text='下一页', command=self.change).pack()
        btn_back = tk.Button(self.face1, text='上一页', command=self.back).pack()

    def change(self):
        self.face1.destroy()
        face_shouhuo(self.master)

    def back(self):
        self.face1.destroy()
        face_jieguo(self.master)


class face_shouhuo():
    def __init__(self, master):
        self.master = master
        self.face1 = tk.Frame(self.master, )
        self.face1.pack()
        label1 = tk.Label(self.face1, text='收获', font=('Arial', 16)).pack()
        btn_back = tk.Button(self.face1, text='上一页', command=self.back).pack()

    def back(self):
        self.face1.destroy()
        face_zhanwang(self.master)


if __name__ == '__main__':
    root = tk.Tk()
    basedesk(root)
    root.mainloop()
