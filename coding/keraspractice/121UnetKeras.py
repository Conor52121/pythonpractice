# -*- coding: utf-8 -*-
# @Time: 2020/6/30 15:26
# @Author: wangshengkang
# @Software: PyCharm
# -----------------------------------代码布局----------------------------------------------
# 1导入相关包
# 2降采样部分
# 3上采样部分
# 4Unet
# -----------------------------------代码布局----------------------------------------------
# -----------------------------------1导入相关包----------------------------------------------
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model


# -----------------------------------1导入相关包----------------------------------------------
# -----------------------------------2降采样部分----------------------------------------------
def downsampling_block(input_tensor, filters, padding='valid', batchnorm=False, dropout=0.0):
    _, height, width, _ = K.int_shape(input_tensor)  # 获取高度和宽度
    assert height % 2 == 0
    assert width % 2 == 0

    # 两层卷积操作
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(input_tensor)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return MaxPooling2D(pool_size=(2, 2))(x), x  # 返回的是池化后的值和dropout后的值，这里dropout后的值用于上采样特征级联


# -----------------------------------2降采样部分----------------------------------------------
# -----------------------------------3上采样部分----------------------------------------------
def upsampling_block(input_tensor, skip_tensor, filters, padding='valid', batchnorm=False, dropout=0.0):
    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2))(input_tensor)
    _, x_height, x_width, _ = K.int_shape(x)
    _, s_height, s_width, _ = K.int_shape(skip_tensor)
    h_crop = s_height - x_height
    w_crop = s_width - x_width
    assert h_crop >= 0
    assert w_crop >= 0
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    else:  # 使级联时像素大小一致
        cropping = ((h_crop // 2, h_crop - h_crop // 2), (w_crop // 2, w_crop - w_crop // 2))
        y = Cropping2D(cropping=cropping)(skip_tensor)

    x = Concatenate()([x, y])  # 特征级联

    # 两层卷积操作
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return x  # 返回dropout后的值


# -----------------------------------3上采样部分----------------------------------------------
# -----------------------------------4Unet----------------------------------------------
def unet(height, width, channels, classes, features=64, depth=4,
         temperature=1.0, padding='valid', batchnorm=False, dropout=0.0):
    x = Input(shape=(height, width, channels))  # 输入的特征
    inputs = x

    skips = []  # 用于存放下采样中，每个深度后，dropout后的值，以供之后级联使用
    for i in range(depth):
        x, x0 = downsampling_block(x, features, padding, batchnorm, dropout)
        skips.append(x0)
        features *= 2  # 下采样过程中，每个深度往下，特征翻倍，即每次使用翻倍数目的滤波器

    x = Conv2D(filters=features, kernel_size=(3, 3,), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        features //= 2  # 每个深度往上。特征减少一倍
        x = upsampling_block(x, skips[i], features, padding, batchnorm, dropout)

    x = Conv2D(filters=classes, kernel_size=(1, 1))(x)

    logits = Lambda(lambda z: z / temperature)(x)  # 简单的对x做一个变换
    probabilities = Activation('softmax')(logits)  # 对输出的两类做softmax，转换为概率。形式如【0.1,0.9],则预测为第二类的概率更大。

    return Model(inputs, outputs=probabilities)


# -----------------------------------4Unet----------------------------------------------

model = unet(572, 572, 3, 10)
model.summary()
plot_model(model, to_file='unet.png', show_shapes=True)
