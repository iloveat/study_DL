# -*- coding: utf-8 -*-
from mxnet import image
from mxnet import nd
import numpy as np


def transform_train(data, label):
    im = data.astype('float32') / 255
    aug_list = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0,
                                     rand_crop=False, rand_resize=False, rand_mirror=True,
                                     mean=np.array([0.4914, 0.4822, 0.4465]),
                                     std=np.array([0.2023, 0.1994, 0.2010]),
                                     brightness=0, contrast=0,
                                     saturation=0, hue=0,
                                     pca_noise=0, rand_gray=0, inter_method=2)
    for aug in aug_list:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).asscalar().astype('float32')


# 测试时，无需对图像做标准化以外的增强数据处理
def transform_test(data, label):
    im = data.astype('float32') / 255
    aug_list = image.CreateAugmenter(data_shape=(3, 32, 32),
                                     mean=np.array([0.4914, 0.4822, 0.4465]),
                                     std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in aug_list:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).asscalar().astype('float32')