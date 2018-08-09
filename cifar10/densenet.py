# -*- coding: utf-8 -*-
from mxnet.gluon import nn
from mxnet import init
import math


"""
Densenet
https://github.com/SherlockLiao/cifar10-gluon/blob/master/densenet.py
https://github.com/yinglang/CIFAR10_mxnet/blob/master/netlib.py
"""


class Bottleneck(nn.HybridBlock):
    def __init__(self, growth_rate):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels=inter_channels, kernel_size=1, use_bias=False,
                                   weight_initializer=init.Normal(math.sqrt(2. / inter_channels)))
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels=growth_rate, kernel_size=3, padding=1, use_bias=False,
                                   weight_initializer=init.Normal(math.sqrt(2. / (9 * growth_rate))))

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = F.concat(* [x, out], dim=1)
        return out


class SingleLayer(nn.HybridBlock):
    def __init__(self, growth_rate):
        super(SingleLayer, self).__init__()
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels=growth_rate, kernel_size=3, padding=1, use_bias=False,
                                   weight_initializer=init.Normal(math.sqrt(2. / (9 * growth_rate))))

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.concat(x, out, dim=1)
        return out


class Transition(nn.HybridBlock):
    def __init__(self, out_channels):
        super(Transition, self).__init__()
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels=out_channels, kernel_size=1, use_bias=False,
                                   weight_initializer=init.Normal(math.sqrt(2. / out_channels)))

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.Pooling(out, kernel=(2, 2), stride=(2, 2), pool_type='avg')
        return out


class DenseNet(nn.HybridBlock):
    def __init__(self, growth_rate, depth, reduction, n_classes, bottleneck):
        super(DenseNet, self).__init__()

        n_dense_blocks = (depth - 4) // 3
        if bottleneck:
            n_dense_blocks //= 2

        n_channels = 2 * growth_rate
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=n_channels, kernel_size=3, padding=1, use_bias=False,
                                   weight_initializer=init.Normal(math.sqrt(2. / n_channels)))
            self.dense1 = self.make_dense(growth_rate, n_dense_blocks, bottleneck)

        n_channels += n_dense_blocks * growth_rate
        out_channels = int(math.floor(n_channels * reduction))
        with self.name_scope():
            self.trans1 = Transition(out_channels)

        n_channels = out_channels
        with self.name_scope():
            self.dense2 = self.make_dense(growth_rate, n_dense_blocks, bottleneck)
            
        n_channels += n_dense_blocks * growth_rate
        out_channels = int(math.floor(n_channels * reduction))
        with self.name_scope():
            self.trans2 = Transition(out_channels)

        n_channels = out_channels
        with self.name_scope():
            self.dense3 = self.make_dense(growth_rate, n_dense_blocks, bottleneck)
        n_channels += n_dense_blocks * growth_rate

        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.fc = nn.Dense(n_classes)

    def make_dense(self, growth_rate, n_dense_blocks, bottleneck):
        layers = nn.HybridSequential()
        for i in range(int(n_dense_blocks)):
            if bottleneck:
                layers.add(Bottleneck(growth_rate))
            else:
                layers.add(SingleLayer(growth_rate))
        return layers

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.Pooling(F.relu(self.bn1(out)), global_pool=1, pool_type='avg', kernel=(8, 8))
        out = self.fc(out)
        return out





