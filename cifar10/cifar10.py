# -*- coding: utf-8 -*-
import augment_data as agd
import resnet as rsn
import densenet as dsn
import utils as utils
from mxnet import autograd
from mxnet import gluon
from mxnet import init
from mxnet import nd
import pandas as pd
import datetime


fine_tune = False
prev_params_file = '../model/cifar10/cifar10-0033.params'


demo = False
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
valid_ratio = 0.1


if demo:
    data_dir = '../data/cifar10'
    train_dir = 'train_tiny'
    test_dir = 'test_tiny'
    batch_size = 1
else:
    data_dir = '/home/brycezou/DATA/cifar10'
    train_dir = 'train'
    test_dir = 'test'
    batch_size = 128


input_str = data_dir + '/' + input_dir + '/'

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = gluon.data.vision.ImageFolderDataset(input_str + 'train', flag=1, transform=agd.transform_train)
valid_ds = gluon.data.vision.ImageFolderDataset(input_str + 'valid', flag=1, transform=agd.transform_test)
test_ds = gluon.data.vision.ImageFolderDataset(input_str + 'test', flag=1, transform=agd.transform_test)
train_valid_ds = gluon.data.vision.ImageFolderDataset(input_str + 'train_valid', flag=1, transform=agd.transform_train)

train_data = gluon.data.DataLoader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = gluon.data.DataLoader(valid_ds, batch_size, shuffle=True, last_batch='keep')
test_data = gluon.data.DataLoader(test_ds, batch_size, shuffle=False, last_batch='keep')
train_valid_data = gluon.data.DataLoader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    prev_time = datetime.datetime.now()

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        """
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        """
        if epoch in [90, 140]:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)

        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_accuracy += utils.accuracy(output, label)

        curr_time = datetime.datetime.now()
        h, remainder = divmod((curr_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data), train_accuracy / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data), train_accuracy / len(train_data)))

        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        prev_time = curr_time

        net.save_params('../model/cifar10/cifar10-%04d.params' % epoch)


ctx = utils.try_gpu()
num_epochs = 200
learning_rate = 0.1
weight_decay = 5e-4
lr_period = 90
lr_decay = 0.1


def get_net(ctx):
    num_outputs = 10
    net = rsn.ResNet_18(num_outputs)
    # net = dsn.DenseNet(growth_rate=12, depth=100, reduction=0.5, bottleneck=True, n_classes=10)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net

if fine_tune:
    net = get_net(ctx=ctx)
    net.hybridize()
    net.load_params(prev_params_file, ctx=ctx)
else:
    net = get_net(ctx)
    net.hybridize()

train(net, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)


preds = []
for data, label in test_data:
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)



