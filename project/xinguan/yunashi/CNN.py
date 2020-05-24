from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation

img = "0.jpg"

def CNN_model():
    model = Sequential()
    # 第一层CNN
    # 第一个参数是卷积核的数量，第二三个参数是卷积核的大小
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    # 第二层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    # 第三层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第四层CNN
    model.add(Convolution2D(9, 3, 3, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model










