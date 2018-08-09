# -*- coding: utf-8 -*-
from mxnet import nd
from mxnet.gluon import nn


def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors*4, 3, padding=1)


"""
num_anchors = 5

box_pred = box_predictor(num_anchors=num_anchors)
box_pred.initialize()
x = nd.zeros((7, 3, 20, 20))
y = box_pred(x)

# NOTE: 需要基于每个anchor预测一组坐标(4个左标点)
# 第7个样本上,在以(0,0)为中心的第a个anchor上预测得到的坐标
a = 4
print y[6, :, 0, 0][a*4:a*4+4]
print y.shape
"""


