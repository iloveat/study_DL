# -*- coding: utf-8 -*-
import mxnet_bisru as rnn
import datetime
import time
import mxnet as mx
from mxnet import gluon
from mxnet import init
from mxnet import autograd
from mxnet import nd
import sys
sys.path.append('../util')
import path_provider as pap
import data_provider as dap


def data_iter(data_reader):
    while not data_reader.is_finish():
        train_set_x, train_set_y = data_reader.load_one_sample()
        yield nd.array(train_set_x), nd.array(train_set_y)


dir_base = '/home/brycezou/DATA/Audio/EM2000/feature_16k/'
b_16k = True
num_train_file = 1600
num_valid_file = 399
check_point = 9
fine_tune = False


if b_16k:
    n_ins = 87
    n_outs = 187
else:
    n_ins = 87
    n_outs = 193
dir_lab_norm = 'nn_no_silence_lab_norm_'+str(n_ins)
dir_cmp_norm = 'nn_norm_mgc_lf0_vuv_bap_'+str(n_outs)
file_names = 'file_id_list_full.scp'


ctx = mx.gpu()
net = rnn.MXNET_BiSRU(num_out=n_outs, num_hidden=512)
net.initialize(ctx=ctx, init=init.Xavier())
net.hybridize()
save_model_prefix = dir_base+'models/mxnet_bisru_'
training_log_file = dir_base+'train_mxnet_bisru.log'
if fine_tune:
    net.load_params(save_model_prefix + '%d.params' % check_point, ctx=ctx)

learning_rate = 0.002
num_epochs = 100
batch_size = 1
train_schedule = [90, 150]
L2loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'momentum': 0.9, 'wd': 5e-4})


# prepare the training data
file_list = pap.read_file_list(dir_base+file_names)
cmp_norm_file_list = pap.prepare_file_path_list(file_list, dir_base+dir_cmp_norm, '.cmp')
lab_norm_file_list = pap.prepare_file_path_list(file_list, dir_base+dir_lab_norm, '.lab')

train_x_file_list = lab_norm_file_list[0:num_train_file]
train_y_file_list = cmp_norm_file_list[0:num_train_file]
valid_x_file_list = lab_norm_file_list[num_train_file:num_train_file+num_valid_file]
valid_y_file_list = cmp_norm_file_list[num_train_file:num_train_file+num_valid_file]

train_data_reader = dap.ListDataProvider(x_file_list=train_x_file_list, y_file_list=train_y_file_list,
                                         n_ins=n_ins, n_outs=n_outs, shuffle=True)
valid_data_reader = dap.ListDataProvider(x_file_list=valid_x_file_list, y_file_list=valid_y_file_list,
                                         n_ins=n_ins, n_outs=n_outs, shuffle=False)
train_data_reader.reset()
valid_data_reader.reset()


for epoch in range(num_epochs):
    begin_time = time.time()
    log_file = open(training_log_file, 'a')

    # train
    train_loss = 0.0
    for data, label in data_iter(train_data_reader):
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = L2loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
    train_data_reader.reset()

    # validation
    valid_loss = 0.0
    for data, label in data_iter(valid_data_reader):
        output = net(data.as_in_context(ctx))
        loss = L2loss(output, label.as_in_context(ctx))
        valid_loss += nd.mean(loss).asscalar()
    valid_data_reader.reset()

    # save model
    net.save_params(save_model_prefix + '%d.params' % epoch)

    # calculate loss
    curr_train_loss = train_loss / train_data_reader.list_size
    curr_valid_loss = valid_loss / valid_data_reader.list_size

    # update learning rate if possible
    # if epoch in train_schedule:
    #     trainer.set_learning_rate(trainer.learning_rate * 0.5)

    curr_time = datetime.datetime.now().strftime('%Y-%m-%d %T')
    message = 'Epoch:%04d, Lr:%f, Train:%f, Valid:%f, Time:%f, Time:%s' \
              % (epoch, trainer.learning_rate, curr_train_loss, curr_valid_loss, time.time()-begin_time, curr_time)
    log_file.write(message+'\n')
    log_file.close()
    print message



















