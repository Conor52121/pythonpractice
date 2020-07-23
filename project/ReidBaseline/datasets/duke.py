# -*- coding: utf-8 -*-
# @Time: 2020/7/23 8:07
# @Author: wangshengkang
# @Software: PyCharm
# 此文件仿照prepare.py对duke数据集进行读取
import os
from shutil import copyfile
import argparse
'''
行人重识别(ReID) ——数据集描述 DukeMTMC-reID

数据集简介
　　DukeMTMC 数据集是一个大规模标记的多目标多摄像机行人跟踪数据集。它提供了一个由 8 个
同步摄像机记录的新型大型高清视频数据集，具有 7,000 多个单摄像机轨迹和超过 2,700 多个
独立人物，DukeMTMC-reID 是 DukeMTMC 数据集的行人重识别子集，并且提供了人工标注的
bounding box。

目录结构
DukeMTMC-reID
　　├── bounding_box_test
　　　　　　　├── 0002_c1_f0044158.jpg
　　　　　　　├── 3761_c6_f0183709.jpg
　　　　　　　├── 7139_c2_f0160815.jpg
　　├── bounding_box_train
　　　　　　　├── 0001_c2_f0046182.jpg
　　　　　　　├── 0008_c3_f0026318.jpg
　　　　　　　├── 7140_c4_f0175988.jpg
　　├── query
　　　　　　　├── 0005_c2_f0046985.jpg
　　　　　　　├── 0023_c4_f0031504.jpg
　　　　　　　├── 7139_c2_f0160575.jpg
　　└── CITATION_DukeMTMC.txt
　　└── CITATION_DukeMTMC-reID.txt
　　└── LICENSE_DukeMTMC.txt
　　└── LICENSE_DukeMTMC-reID.txt
　　└── README.md

目录介绍
从视频中每 120 帧采样一张图像，得到了 36,411 张图像。一共有 1,404 个人出现在大于两个摄像头下，有 408 个人 (distractor ID) 只出现在一个摄像头下。
1） “bounding_box_test”——用于测试集的 702 人，包含 17,661 张图像（随机采样，702 ID + 408 distractor ID）
2） “bounding_box_train”——用于训练集的 702 人，包含 16,522 张图像（随机采样）
3） “query”——为测试集中的 702 人在每个摄像头中随机选择一张图像作为 query，共有 2,228 张图像

命名规则
以 0001_c2_f0046182.jpg 为例
1） 0001 表示每个人的标签编号；
2） c2 表示来自第二个摄像头(camera2)，共有 8 个摄像头；
3） f0046182 表示来自第二个摄像头的第 46182 帧。
'''


# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description='Training')
# 调用add_argument()方法添加参数
parser.add_argument('--download_path',
                    default='/data2/wangshengkang/ingenious/a/skillful/datasets/DukeMTMC-reID',
                    type=str,
                    help='dataset path')

# 使用parse_args()解析添加的参数
opt = parser.parse_args()

# 数据集路径
download_path = opt.download_path
#download_path = 'D:\datasets\DukeMTMC-reID\DukeMTMC-reID'

# 如果数据集地址不对，提示
if not os.path.isdir(download_path):
    print('please change the download_path')

# 数据处理后的保存地址
save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)  # 如果没有这个文件夹，创建一个新的
# --------------------------------------------------------------------------------
# query
query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)  # 如果没有这个文件夹，创建一个新的

# os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
# os.walk() 方法是一个简单易用的文件、目录遍历器，可以帮助我们高效的处理文件、目录方面的事情。
# top -- 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。
# root 所指的是当前正在遍历的这个文件夹的本身的地址
# dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
# files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
# topdown --可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。
# 如果 topdown 参数为 True，walk 会遍历top文件夹，与top 文件夹中每一个子目录。
for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':  # 如果后缀名不是jpg，跳出本次循环，不执行余下代码，继续下一次循环
            continue
        ID = name.split('_')  # 图片id，分成四块
        src_path = query_path + '/' + name  # 图片地址
        dst_path = query_save_path + '/' + ID[0]  # 用标签作为地址
        if not os.path.isdir(dst_path):  # 每个label文件夹的第一张图片时会创建新的文件夹，后面的就不用了
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)  # 将原始数据集复制到以label命名的新的文件夹



# -----------------------------------------------------------------------------------
# gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# -------------------------------------------------------------------------------------
# train_all
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# --------------------------------------------------------------------------------------
# train
# val
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        # 每个label的第一张图片会创建val新的文件夹，并且复制到val中去
        # 如果不是第一张图片则会直接跳过if not，复制到train中去
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  # first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
