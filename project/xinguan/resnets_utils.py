# resnets_utils.py
import glob
import os,sys
import numpy as np
import tensorflow as tf
import h5py
import math
import skimage.io as io
from skimage import data_dir
import numpy as np
from networkx.tests.test_convert_pandas import pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def load_dataset():

    if True:
        #数据增强
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            # brightness_range=[0.1, 10],
            horizontal_flip=True,  # 进行随机水平翻转
            vertical_flip=True,  # 进行随机竖直翻转
            fill_mode='nearest')

        # 非新冠肺炎训练数据集数据增强
        path = "TrainData/nonCOVID-traindata/";
        dirs = os.listdir(path)
        nonTraindata = []
        for file in dirs:
            img = load_img(path + file)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)
            nonTraindata.append(x)
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir='TrainData/nonCOVID-traindata/',
                                      save_prefix='0', save_format='jpg'):
                i += 1
                if i > 50:  # 数据扩充倍数，此处为数据扩充50倍
                    break  # 否则生成器会退出循环

        # 新冠肺炎训练数据增强
        path = "TrainData/COVID-traindata/";
        dirs = os.listdir(path)
        Traindata = []
        for file in dirs:
            img = load_img(path + file)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)
            Traindata.append(x)
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir='TrainData/COVID-traindata/',
                                      save_prefix='1', save_format='jpg'):
                i += 1
                if i > 50:  # 数据扩充倍数，此处为数据扩充50倍
                    break  # 否则生成器会退出循环

        # 提取非新冠肺炎训练数据集标签
        path = "TrainData/nonCOVID-traindata/";
        dirs = os.listdir(path)
        nonCOVIDtraindata = []
        for file in dirs:  ## 从训练集中获取图片的文件名
            # print('非新冠训练数据集')
            # print(file)
            label = int(file.split("_")[0])
            nonCOVIDtraindata.append(label)
        print('非新冠训练数据集大小：')
        # print(nonCOVIDValdata)
        print(nonCOVIDtraindata.__len__())

        # 提取新冠训练数据集标签
        path = "TrainData/COVID-traindata/";
        dirs = os.listdir(path)
        COVIDtraindata = []
        for file in dirs:  ## 从训练集中获取图片的文件名
            print(file)
            label = int(file.split("_")[0])
            COVIDtraindata.append(label)
        print('新冠训练数据集大小：')
        # print(nonCOVIDValdata)
        print(COVIDtraindata.__len__())

        # 训练数据集及其标签
        Train_NONimages = sorted(glob.glob('TrainData/NonCOVID-traindata/*.jpg'))  # 加载训练数据
        Train_COVIDimages = sorted(glob.glob('TrainData/COVID-traindata/*.jpg'))
        Train_images = Train_NONimages + Train_COVIDimages
        y = nonCOVIDtraindata + COVIDtraindata

        #filepath = "D:\SoftWare\Pycharm\COVID-DATA\DATA\TrainData.csv"  # 加载训练数据的标签
        #df = pd.read_csv(filepath)
        #y = df.values  # A numpy array containing age label of 559 persons
        #y = y[:, 1]


        Val_images = sorted(glob.glob('TestData\Valdata\*.jpg'))  # 加载训练数据
        path = "TestData\ValData.csv"  # 加载训练数据的标签
        dft = pd.read_csv(path)
        yt = dft.values
        yt = yt[:, 1]


    # 将训练数据集与其标签相对应
        np.random.seed(9)
        np.random.shuffle(Train_images)
        np.random.seed(9)
        np.random.shuffle(y)
    # print(Train_images)
    # print(y)
        np.random.seed(9)
        np.random.shuffle(Val_images)
        np.random.seed(9)
        np.random.shuffle(yt)
    # print(Val_images)
    # print(yt)
    #train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = Train_images
    train_set_y_orig = y
    #train_set_x_orig = np.array(Train_images[:])  # your train set features
    #train_set_y_orig = np.array(y[:])  # your train set labels
    #print(train_set_x_orig)
    #print(train_set_x_orig.__len__())
    #print(train_set_y_orig)
    #print(train_set_y_orig.__len__())

    #test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = Val_images
    test_set_y_orig = yt
    #test_set_x_orig = np.array(Val_images[:])  # your test set features
    #test_set_y_orig = np.array(yt[:])  # your test set labels

    #classes = np.array(y["list_classes"][:])  # the list of classes

    #train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    #print(train_set_y_orig)
    #test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def random_mini_batches(X, Y, mini_batch_size=2, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

    def convert_to_one_hot(Y, C):
        Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

    return Z3


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction
