# -*- coding: utf-8 -*-
from mxnet import nd


aa = nd.zeros(shape=(3, 5))
print aa
bb = nd.ones(shape=(3, 5))
print bb
cc = nd.zeros_like(aa)
print id(cc)
print cc
dd = nd.ones(shape=(5,))
dd[:] = 9
print dd.shape
cc[2][:] = dd
print id(cc)
print cc
ee = nd.concat(aa, bb, dim=1)
ee[1, 7] = 123
print ee
print nd.concat(aa, dd.reshape(shape=(-1, 5)), dim=0)









