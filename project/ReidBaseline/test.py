# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='5', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='/data2/wangshengkang/ingenious/a/skillful/datasets/Market-1501-v15.09.15/pytorch', type=str,
                    help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
# 使用parse_args()解析添加的参数
opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)

opt.stride = config['stride']

if 'nclasses' in config:  # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751

str_ids = opt.gpu_ids.split(',')  # 将gpu字符串分开
# which_epoch = opt.which_epoch
name = opt.name  # 模型名字
test_dir = opt.test_dir  # 测试集地址

gpu_ids = []  # 创建gpu列表
for str_id in str_ids:
    id = int(str_id)  # 将str转为int
    if id >= 0:
        gpu_ids.append(id)  # 将可用gpu加入gpu列表

print('We use the scale: %s' % opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids) > 0:  # 如果有gpu
    torch.cuda.set_device(gpu_ids[0])  # 设置用哪块gpu，只用第一块就够了
    '''
    总的来说，大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    一般来讲，应该遵循以下准则：
    如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
    如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。   
    '''
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir  # 数据集地址
if data_dir == 'market':
    data_dir = '/data2/wangshengkang/ingenious/a/skillful/datasets/Market-1501-v15.09.15/pytorch'
elif data_dir == 'duke':
    data_dir = '/data2/wangshengkang/ingenious/a/skillful/datasets/DukeMTMC-reID/pytorch'
elif data_dir == 'msmt':
    data_dir = '/data2/wangshengkang/ingenious/a/skillful/datasets/MSMT17/pytorch'

# 数据集弄成dataloader的形式
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in ['gallery', 'query']}
class_names = image_datasets['query'].classes  # 行人ID的数量
use_gpu = torch.cuda.is_available()  # 是否有gpu


######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = os.path.join('./model', name, 'net_%s.pth' % opt.which_epoch)  # 保存模型的路径
    network.load_state_dict(torch.load(save_path))  # 保存模型
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    #arange
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# 提取特征
def extract_feature(model, dataloaders):
    features = torch.FloatTensor()#创建一个tensor
    count = 0#目前所有batch的图片数量初始化
    for data in dataloaders:
        img, label = data  # 图片和标签
        n, c, h, w = img.size()  # 图片的N,C,H,W
        count += n  # 将每个batch的图片数量加起来
        print(count)  # 打印目前所有batch的图片数量
        ff = torch.FloatTensor(n, 512).zero_().cuda()#创建一个大小为n*512的0矩阵

        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())  # 放到gpu里面，并且用varibale包装
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic',
                                                          align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


gallery_path = image_datasets['gallery'].imgs  # gallery数据集
query_path = image_datasets['query'].imgs  # query数据集

gallery_cam, gallery_label = get_id(gallery_path)  # 获取gallery的id
query_cam, query_label = get_id(query_path)  # 获取query的id

######################################################################
# Load Collected data Trained model
print('-------test-----------')

# 调用模型，resnet
model_structure = ft_net(opt.nclasses, stride=opt.stride)

# 加载模型参数
model = load_network(model_structure)

# Remove the final fc layer and classifier layer
# 讲模型最后的分类层去掉
model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()  # 如果有gpu，模型放到gpu里面

# Extract feature
# 用于停止autograd模块的工作，以起到加速和节省显存的作用，具体行为就是
# 停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm
# 层的行为。
with torch.no_grad():
    # 提取gallery特征
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    # 提取query特征
    query_feature = extract_feature(model, dataloaders['query'])

# Save to Matlab for check
result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
scipy.io.savemat('pytorch_result.mat', result)

print(opt.name)  # 打印此时运行程序的名字
result = './model/%s/result.txt' % opt.name
# | 表示管道，上一条命令的输出，作为下一条命令参数
# Linux tee命令用于读取标准输入的数据，并将其内容输出成文件。
# tee指令会从标准输入设备读取数据，将其内容输出到标准输出设备，同时保存成文件。
# -a或--append 　附加到既有文件的后面，而非覆盖它。
os.system('python evaluate_gpu.py | tee -a %s' % result)#将运行结果存放到result里面
