# -*- coding: utf-8 -*-
# @Time: 2020/7/23 8:54
# @Author: wangshengkang
# @Software: PyCharm

import os
from shutil import copyfile
import argparse
'''
CVPR2018会议上，提出了一个新的更接近真实场景的大型数据集MSMT17，即Multi-Scene Multi-Time，涵盖了多场景多时段。
MSMT17数据集描述
        数据集采用了安防在校园内的15个摄像头网络，其中包含12个户外摄像头和3个室内摄像头。为了采集原始监控视频，在一个月里选择了具有不同天气条件的4天。每天采集3个小时的视频，涵盖了早上、中午、下午三个时间段。因此，总共的原始视频时长为180小时。
MSMT17数据集的特点如下：
（1）数目更多的行人、包围框、摄像头数；
（2）复杂的场景和背景；
（3）涵盖多时段，因此有复杂的光照变化；
（4）更好的行人检测器（faster RCNN）
评估协议        按照训练-测试为1：3的比例对数据集进行随机划分，而不是像其他数据集一样均等划分。这样做的目的是鼓励高效率的训练策略，由于在真实应用中标注数据的昂贵。
        最后，训练集包含1041个行人共32621个包围框，而测试集包括3060个行人共93820个包围框。对于测试集，11659个包围框被随机选出来作为query，而其它82161个包围框作为gallery.
        测试指标为CMC曲线和mAP. 对于每个query, 可能存在多个正匹配。

目录结构
MSMT17
├── bounding_box_test
　　　　　　　├── 0000_c1_0002.jpg
　　　　　　　├── 0000_c1_0003.jpg
　　　　　　　├── 0000_c1_0005.jpg
├── bounding_box_train
　　　　　　　├── 0000_c1_0000.jpg
　　　　　　　├── 0000_c1_0001.jpg
　　　　　　　├── 0000_c1_0002.jpg
├── query
　　　　　　　├── 0000_c1_0000.jpg
　　　　　　　├── 0000_c1_0001.jpg
　　　　　　　├── 0000_c14_0030.jpg
'''

# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description='Training')
# 调用add_argument()方法添加参数
parser.add_argument('--download_path',
                    default='/data2/wangshengkang/ingenious/a/skillful/datasets/MSMT17',
                    type=str,
                    help='dataset path')

# 使用parse_args()解析添加的参数
opt = parser.parse_args()

# 数据集路径
download_path = opt.download_path


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
