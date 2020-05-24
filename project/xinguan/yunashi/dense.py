import os
import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Convolution2D
from keras.layers import Flatten, Dense, Dropout,BatchNormalization, Activation, Lambda
from keras.regularizers import l2
from keras.layers import Input, Concatenate, concatenate
import keras.backend as K
import tensorflow as tf
from keras.models import Model,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model,np_utils
from keras import regularizers
import cv2


#定义卷积模块
def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    x = Activation('relu')(input)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    return x

#定义dense模块
def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay) #经过卷积
        feature_list.append(x) #将卷积得到的特征与原特征一起输入到下一层
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate# 卷积核个数加上增长率
    return x, nb_filter
