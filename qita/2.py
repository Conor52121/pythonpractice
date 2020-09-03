# -*- coding: utf-8 -*-
# @Time: 2020/8/22 21:22
# @Author: wangshengkang
# @Software: PyCharm

from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from norm import Normalize


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,  # 从torchvision里面下载模型
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_triplet_features=0):
        super(ResNet, self).__init__()

        self.depth = depth  # 模型深度
        self.pretrained = pretrained  # 是否预训练
        self.cut_at_pooling = cut_at_pooling  # 是否切掉最后一个全局池化层，默认为false

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:  # 如果所选的深度没有的话报错提示
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)  # 加载预训练的resnet

        # Fix layers [conv1 ~ layer2]
        fixed_names = []  # 创建需要固定参数的层的列表
        for name, module in self.base._modules.items():
            if name == "layer3":  # layer3层之前的层，参数固定住
                # assert fixed_names == ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
                break
            fixed_names.append(name)  # 将固定参数的层加入到列表中
            for param in module.parameters():
                param.requires_grad = False

        if not self.cut_at_pooling:
            self.num_features = num_features  # 4096
            self.norm = norm  # 默认为False
            self.dropout = dropout  # 0.5
            self.has_embedding = num_features > 0  # true
            self.num_classes = num_classes
            self.num_triplet_features = num_triplet_features

            self.l2norm = Normalize(2)

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:  # true
                self.feat = nn.Linear(out_planes, self.num_features)  # num_features4096
                self.feat_bn = nn.BatchNorm1d(self.num_features)  # num_features4096
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout >= 0:  # true
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:  # true
                self.classifier = nn.Linear(self.num_features, self.num_classes)  # 分类器
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        if not self.pretrained:  # 如果没有预训练，初始化参数
            self.reset_params()
    # 128*3*256*128
    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':  # 到avgpool停止
                break
            else:
                x = module(x)  # avgpool之前的层都使用

        if self.cut_at_pooling:  # 如果切掉the last global pooling layer，直接返回x并退出
            return x
        # -----------------------Pooling-5层之前的被称为base network------------------------------------
        x = F.avg_pool2d(x, x.size()[2:])  # 这是论文中所说的Pooling-5层
        x = x.view(x.size(0), -1)

        if output_feature == 'pool5':  # 如果在测试的时候
            x = F.normalize(x)  # Pooling-5层出来的特征经过L2-normalized后输出作为image feature
            return x

        if self.has_embedding:  # true
            x = self.feat(x)  # 全连接层  这就是论文中的FC-4096
            x = self.feat_bn(x)  # batch normalization
            tgt_feat = F.normalize(x)
            tgt_feat = self.drop(tgt_feat)#We use dropout to avoid overfitting. Without dropout, the results may slightly lower.
            if output_feature == 'tgt_feat':  # 在target域的时候在这里就停止
                return tgt_feat
        if self.norm:  # 默认为False
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)  # ReLU
        if self.dropout > 0:
            x = self.drop(x)  # Dropout
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

from torchsummary import summary
net=resnet50()
print(net)
summary(net,(3,256,128))
