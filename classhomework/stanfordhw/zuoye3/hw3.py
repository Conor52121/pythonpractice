import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time


# 当label等于false时即不需要返回label，用来处理test集，方便统一管理
# 利用 OpenCV (cv2) 讀入照片並存放在 numpy array 中
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))  # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。sorted()对所有可迭代的对象进行排序操作
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)  # zeros()返回来一个给定形状和类型的用0填充的数组。uint8无符号整数（0 to 255）
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))  # x为128*128*3
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x

print("Reading data")
train_x, train_y = readfile("C:/reiddatasets/food-11/food-11/training", True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile("C:/reiddatasets/food-11/food-11/validation", True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile("C:/reiddatasets/food-11/food-11/testing", False)
print("Size of Testing data = {}".format(len(test_x)))

# 对train和validation进行数据增强(data augmentation),可参考链接https://blog.csdn.net/lanmengyiyu/article/details/79658545
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随即将图片水平翻转
    transforms.RandomRotation(15),  # 随即旋转图片15度
    transforms.ToTensor(),  # 将图片转成 Tensor
    # transforms.Normalize([0.34361264, 0.45097706, 0.5550434], [0.2783836, 0.27154413, 0.26915196])  # Normalize
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),  # 将图片转成 Tensor
    # transforms.Normalize([0.35057753, 0.45622048, 0.5620281], [0.2769976, 0.27097332, 0.26702392])  # Normalize
])

# 集成了一个 Dataset类之后，我们需要重写 len 方法，该方法提供了dataset的大小；
# getitem 方法， 该方法支持从 0 到 len(self)的索引
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:  # 如果没有标签那么只返回X
            return X

# Dataset是一个包装类，用来将数据包装为Dataset类，
# 然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。
batch_size = 50
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # 进行shuffle随机
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

"""定义模型以及设置其中架构和一些参数"""


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            # 第一层卷积层和池化层的配置
            # 第一个参数代表输入数据的通道数，例RGB图片通道数为3；
            # 第二个参数代表输出数据的通道数，这个根据模型调整；
            # 第三个参数是卷积核大小
            # 第四个参数是stride，步长
            # 第五个参数是padding，补1
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]，即输出是64*128*128维的图片
            nn.BatchNorm2d(64),  # Normalize
            nn.ReLU(),  # activate函数是relu函数
            # 第一个参数是kernel_size，max pooling的窗口大小，
            # 第二个参数是stride，max pooling的窗口移动的步长。默认值是kernel_size
            # 第三个参数输入的每一条边补充0的层数，默认是0
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            # 第二层卷积层和池化层的配置
            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            # 第三层卷积层和池化层的配置
            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            # 第四层卷积层和池化层的配置
            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            # 第五层卷积层和池化层的配置
            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)  # 50*512*4*4
        out = out.view(out.size()[0], -1)  # 拉伸为一维 50*8192
        return self.fc(out)  # 变为11个类别

"""# Training

使用training set訓練，並使用validation set尋找好的參數
"""

model = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss 正交熵
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer 使用 Adam，学习率是0.1
num_epoch = 30  # 迭代30次

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda())  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda())  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()  # 评估模式
    with torch.no_grad():  # 防止GPU爆
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

"""得到好的參數後，我們使用training set和validation set共同訓練（資料量變多，模型效果較好）"""

train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())  # 预测的结果
        batch_loss = loss(train_pred, data[1].cuda())  # data[1]代表真实值
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        # 將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
          (epoch + 1, num_epoch, time.time() - epoch_start_time, \
           train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))


"""# Testing
利用剛剛 train 好的 model 進行 prediction
"""

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# 將結果寫入 csv 檔
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))