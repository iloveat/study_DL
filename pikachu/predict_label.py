# -*- coding: utf-8 -*-
from mxnet import nd
from mxnet.gluon import nn


def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    """输出的通道数是num_anchors*(num_classes+1),每个通道对应一个锚框对某个类的置信度"""
    return nn.Conv2D(channels=num_anchors*(num_classes+1), kernel_size=3, padding=1)


"""
num_classes = 1
num_anchors = 5

cls_pred = class_predictor(num_anchors=num_anchors, num_classes=num_classes)
cls_pred.initialize()
x = nd.zeros((7, 3, 20, 20))
y = cls_pred(x)

# 在第7个样本上,以(0,0)为中心的第a个anchor,
# 只包含背景的置信度: b=0
# 包含第b个物体的置信度: 0 < b <= num_classes
a = 4
b = 1
print y[6, :, 0, 0][a*(num_classes+1)+b]
print y.shape
"""


