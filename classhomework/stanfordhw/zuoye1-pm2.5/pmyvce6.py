import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def wash_train_data():
    raw_train_data = pd.read_csv('train.csv')

    # 将文字下雨文字标注转化为0,1
    raw_train_data.replace({'RAINFALL':1.0,'NR':0.0},inplace=True)

    # 丢掉前面几列文字说明
    raw_train_data.drop(columns=['ら戳','代','代兜'],inplace=True)

    # 转化为浮点型
    raw_train_data = raw_train_data.astype("float")

    # 抽取数据
    length,width = raw_train_data.shape
    dates = int(length / 18)
    new_train_data = np.zeros(shape=(16 * dates,145),dtype=np.float32)
    # 8+1个成一组
    index = 0
    for i in range(dates):
        for j in range(width - 9 + 1):
            input = raw_train_data.values[i*18:(i+1)*18,j:j+8]
            # pm2.5在第10行
            label = raw_train_data.values[i*18+9,j+8]
            # 将结果放在行的后面
            data = np.append(input,label).reshape(1,-1)
            new_train_data[index] = data
            index += 1

    np.savetxt("./new_train_data.csv",new_train_data,delimiter = ',')


def wash_test_data():
    raw_test_data = pd.read_csv('test.csv')

    # 将文字下雨文字标注转化为0,1
    raw_test_data.replace({'RAINFALL': 1.0, 'NR': 0.0}, inplace=True)

    # 丢掉前面几列文字说明
    raw_test_data.drop(columns=['id', 'type'], inplace=True)

    # 转化为浮点型
    raw_train_data = raw_test_data.astype("float")

    # 抽取数据
    length, width = raw_train_data.shape
    dates = int(length / 18)
    new_train_data = np.zeros(shape=(16 * dates, 145), dtype=np.float32)
    # 8+1个成一组
    index = 0
    for i in range(dates):
        for j in range(width - 9 + 1):
            input = raw_train_data.values[i * 18:(i + 1) * 18, j:j + 8]
            # pm2.5在第10行
            label = raw_train_data.values[i * 18 + 9, j + 8]
            # 将结果放在行的后面
            data = np.append(input, label).reshape(1, -1)
            new_train_data[index] = data
            index += 1

    np.savetxt("./new_test_data.csv", new_train_data, delimiter=',')


def load_para():
    index = None
    fileName = None
    for root, dirs, files in os.walk("./data/"):
        for file in files:
            if "weights_and_bias" in file:
                _index = int(file.split(' ')[1])
                if index is None:
                    index = _index
                    fileName = file
                else:
                    if _index > index:
                        index = _index
                        fileName = file
    return fileName,index


def save_para(data):
    _,index = load_para()
    np.savetxt("./data/weights_and_bias {} .csv".format(index + 1), data, delimiter=',')


def main():
    # wash_test_data()
    # wash_train_data()
    data = np.loadtxt("new_train_data.csv", delimiter=",", skiprows=0)
    n_samples,n_dims = data.shape
    # 最后一个维度是labels
    n_dims -= 1
    n_batches = 5
    batch_size = int(n_samples/n_batches)
    save_per_batches = 1000

    # A = np.concatenate((inputs,np.zeros(shape=(n_samples,1))),axis=1)
    # correct_answer = np.linalg.lstsq(A, labels)[0]
    # predict = np.dot(A,correct_answer)
    # delta_y = predict - labels
    # loss = np.mean(delta_y ** 2)
    # print(loss) #36.24
    # print(correct_answer)
    fileName,index = load_para()

    weights = bias = None
    if fileName is not None:
        weights_and_bias = np.loadtxt("./data/"+fileName, delimiter=",", skiprows=0)
        weights = weights_and_bias[:-1]
        bias = weights_and_bias[-1]

    if weights is None:
        # 0.0000000001
        weights = np.random.random(n_dims)
        bias = 0

    losses = []
    dw_square_sum = np.zeros_like(weights)
    db_square_sum = 0
    for i in range(1000):
        np.random.shuffle(data)
        inputs = data[:, 0:-1]
        labels = data[:, -1]
        for j in range(int(n_samples/batch_size)):
            batch_inputs = inputs[j*batch_size:(j+1)*batch_size]
            batch_labels = labels[j*batch_size:(j+1)*batch_size]
            a = np.dot(batch_inputs,weights) + bias
            delta_y = a - batch_labels
            db = delta_y.sum()
            # loss = np.mean(delta_y ** 2)
            dw = np.dot(delta_y,batch_inputs)
            dw_square_sum += dw ** 2
            weights -= 0.000000002*dw
            db_square_sum += db ** 2
            bias -= 0.000000001*db

        a = np.dot(inputs,weights) + bias
        delta_y = a - labels
        loss = np.mean(delta_y ** 2)
        losses.append(loss)
        print("{i}:{loss}".format(i=i,loss=loss))

        if((i+1)%save_per_batches==0):
            save_para(np.append(weights,bias))

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()