# -*- coding: utf-8 -*-
from mxnet import nd
from mxnet.gluon import nn


def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer to halve the feature size"""
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(num_filters, kernel_size=3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    return out


"""
ds = down_sample(10)
ds.initialize()
x = nd.zeros((2, 3, 20, 20))
y = ds(x)
print y.shape
"""


