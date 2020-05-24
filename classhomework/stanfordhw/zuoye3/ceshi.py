import os
import numpy as np
import cv2
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))  # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。sorted()对所有可迭代的对象进行排序操作
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)  # zeros()返回来一个给定形状和类型的用0填充的数组。zeros。uint8无符号整数（0 to 255）
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i : :] = cv2.resize(img, (128, 128))  # 将图片保存为128*128
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x

train_x, train_y = readfile("C:/datasets/ceshi", True)
print("Size of training data = {}".format(len(train_x)))
print(1)
print(train_x)
print(2)
print(train_y)