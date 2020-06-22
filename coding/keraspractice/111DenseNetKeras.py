# -*- coding: utf-8 -*-
# @Time: 2020/6/22 11:27
# @Author: wangshengkang
import numpy as np
import keras
from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
import pickle

data_batch_1=pickle.load(open('cifar-10-batches-py/'))