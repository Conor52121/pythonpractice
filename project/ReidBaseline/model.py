import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels


######################################################################
# 一种权重初始化的方式，由何凯明提出
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []  # 模型list
        if linear:  # 加入linear
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:  # 加入batchnorm
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:  # 加入relu
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:  # 加入dropout
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)  # 序列化方式创建模型
        add_block.apply(weights_init_kaiming)  # 初始化权重

        classifier = []  # 分类器list
        classifier += [nn.Linear(num_bottleneck, class_num)]  # 加入linear
        classifier = nn.Sequential(*classifier)  # 序列化方式创建分类器模型
        classifier.apply(weights_init_classifier)  # 分类器权重初始化

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class ft_net(nn.Module):  # 共有751个行人，分为751类
    def __init__(self, class_num=751, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
        # load the model 加载预训练的restnet50模型
        model_ft = models.resnet50(pretrained=True)
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)  # define our classifier.

    def forward(self, x):
        # 依次调用resnet的各个部分
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)  # use our classifier.
        return x
