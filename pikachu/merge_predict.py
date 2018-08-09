# -*- coding: utf-8 -*-
from mxnet import nd
from predict_label import class_predictor
from down_sample import down_sample


def flatten_prediction(pred):
    return pred.transpose(axes=(0, 2, 3, 1)).flatten()


def concat_predictions(preds):
    return nd.concat(*preds, dim=1)


x = nd.zeros((2, 8, 20, 20))
print 'x      :', x.shape
# 2,8,20,20

cls_pred1 = class_predictor(num_anchors=5, num_classes=10)
cls_pred1.initialize()
y1 = cls_pred1(x)
print 'clss y1:', y1.shape
# 2,55,20,20

ds = down_sample(num_filters=16)
ds.initialize()
x = ds(x)
print 'x      :', x.shape
# 2,16,10,10

cls_pred2 = class_predictor(num_anchors=3, num_classes=10)
cls_pred2.initialize()
y2 = cls_pred2(x)
print 'clss y2:', y2.shape
# 2,33,10,10

flat_y1 = flatten_prediction(y1)
print 'flat y1:', flat_y1.shape

flat_y2 = flatten_prediction(y2)
print 'flat y2:', flat_y2.shape

# 第一个维度是样本个数，不同输出之间是不变。可以将所有输出在第二个维度上拼接起来
y = concat_predictions([flat_y1, flat_y2])
print 'y concat', y.shape



