# import the necessary libraries
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
import random
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_files
from keras import backend as K
from keras.models import model_from_json
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing import image
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, GlobalAveragePooling2D

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.02
session = tf.Session(config=config)
# KTF.set_session(session )

# 输出各个
def print_sizes(x_train, x_valid, x_test):
    # check sizes of dataset
    print('Number of Training images --> %d.' % len(x_train))
    print('Number of Validation images --> %d.' % len(x_valid))
    print('Number of Test images --> %d.' % len(x_test))





def cnn_architecture(input_shape):
    # define the NN architecture
    # input_shape = (28,28,3)
    nn = Sequential()
    nn.add(Conv2D(64,( 5, 5), activation='relu', input_shape=input_shape))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(32, (5, 5), activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Conv2D(16, (5, 5), activation='relu'))
    nn.add(MaxPooling2D(pool_size=(3, 3)))
    nn.add(Flatten())
    # nn.add(GlobalAveragePooling2D())
    nn.add(Dense(700, activation='relu'))
    nn.add(Dropout(0.4))
    nn.add(Dense(600, activation='relu'))
    nn.add(Dense(11, activation='softmax'))
    nn.summary()

    return nn
def accuracy_loss_plots(history):

    # accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('food11_cnn_accuracy.pdf')
    plt.close()
    # loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('food11_cnn_loss.pdf')


def prediction_results(model, x_test, y_test):
    # compute probabilities
    pred_y = model.predict(x_test)
    # assign most probable label
    y_pred = np.argmax(pred_y, axis=1)
    # plot statistics
    print('Analysis of results')
    target_names = ['Bread', 'Dairy_product', 'Dessert', 'Egg', 'Fried_food', 'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Veggies']
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
    print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))






from keras_preprocessing.image import ImageDataGenerator


def main():
    print('Using Keras version', keras.__version__)
    # 读取数据
    train_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, width_shift_range=0.2, height_shift_range=0.2, channel_shift_range=30, rescale=1./255)
    valid_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, width_shift_range=0.2, height_shift_range=0.2, channel_shift_range=30,rescale=1./255)
    # 获取目录路径并生成一批增强数据
    train_generator = train_datagen.flow_from_directory(directory="food11re/training/", target_size=(200, 200), color_mode="rgb", batch_size=120, class_mode="categorical", shuffle=True, seed=25)
    valid_generator = valid_datagen.flow_from_directory(directory="food11re/validation/", target_size=(200, 200), color_mode="rgb", batch_size=120, class_mode="categorical", shuffle=True, seed=25)
    # 输入图片大小
    input_size = (200, 200, 3)
    # 调用模型
    model = cnn_architecture(input_size)
    # 配置参数，sgd优化函数、交叉熵损失函数、metrics: 列表，包含评估模型在训练和测试时的性能的指标
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # 每次的训练大小等于生成的图片数量除以训练批次
    train_size = train_generator.n//train_generator.batch_size
    valid_size = valid_generator.n//valid_generator.batch_size
    # 模型训练
    history = model.fit_generator(generator=train_generator, steps_per_epoch=train_size, validation_data=valid_generator, validation_steps=valid_size, epochs=35)
    # 画图损失函数和准确率
    accuracy_loss_plots(history)
    #模型评估
    score = model.evaluate_generator(generator=valid_generator, steps=len(valid_generator))
    # 打印验证集的损失和准确率
    print('valid loss:', score[0])
    print('valid accuracy:', score[1])
    # 对测试数据进行预测
    Y_pred = model.predict_generator(valid_generator, steps=len(valid_generator), verbose=1)
    # 预测的是y中的最大索引值
    y_pred = np.argmax(Y_pred, axis=1)
    # 获得已有数据的标签
    labels = train_generator.class_indices
    # 便利获得itme标签
    labels = dict((v, k) for k, v in labels.items())
    # 预测的标签是在y_pred中
    predictions = [labels[k] for k in y_pred]

    print('Classification Report')

    target_names = ['Bread', 'Dairy_product', 'Dessert', 'Egg', 'Fried_food', 'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Veggies']
    # 显示分类指标，类别
    print(classification_report(valid_generator.classes, y_pred, target_names=target_names))

    print("finished")


# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin
if __name__ == "__main__":
    main()
