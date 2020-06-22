# -*- coding: utf-8 -*-
# @Time: 2020/6/17 18:25
# @Author: wangshengkang
# -*- coding: utf-8 -*-
# @Time: 2020/6/17 18:13
# @Author: wangshengkang
# %%
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

"""
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
"""
batch_size = 128
seed = 1
epochs = 200
cuda = True
log_interval = 10
h_d = 512
l_d = 32
u_d = 1

torch.manual_seed(seed)
# %% md
### Preparing for the dataset
# %%
# hmnist dataset
import healing_mnist_indep

# %%
hmnist = healing_mnist_indep.HealingMNIST(seq_len=5,  # 5 rotations of each digit
                                          square_count=0,  # 3 out of 5 images have a square added to them
                                          square_size=5,  # the square is 5x5
                                          noise_ratio=0.10,  # on average, 20% of the image is eaten by noise,
                                          digits=range(10),  # only include this digits
                                          test=False
                                          )
# %%
print(hmnist.train_images.shape, hmnist.train_targets.shape)
print(hmnist.train_rotations.shape)
print(hmnist.test_images.shape, hmnist.test_targets.shape)
print(hmnist.test_rotations.shape)

import matplotlib.pyplot as plt
case = 4
fig = plt.figure(figsize=(15, 8))
for i, image in enumerate(hmnist.test_images[case]):
    fig.add_subplot(1, 6, i + 1)
    plt.imshow(image, cmap='gray')
plt.show()
# %%
fig = plt.figure(figsize=(15, 8))
for i, image in enumerate(hmnist.test_targets[case]):
    fig.add_subplot(1, 6, i + 1)
    plt.imshow(image, cmap='gray')
plt.show()
# %%
print(hmnist.test_rotations[case])