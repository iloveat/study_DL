# -*- coding: utf-8 -*-
import numpy as np
import cntk_bisru_fast as rnn
import time
import datetime
import cntk
import sys
sys.path.append('../util')
import path_provider as pap
import data_provider as dap


dir_base = '/home/brycezou/DATA/Audio/EM2000/feature_16k/'
b_16k = True
num_train_file = 1600
num_valid_file = 399


if b_16k:
    n_ins = 87
    n_outs = 187
else:
    n_ins = 87
    n_outs = 193
dir_lab_norm = 'nn_no_silence_lab_norm_'+str(n_ins)
dir_cmp_norm = 'nn_norm_mgc_lf0_vuv_bap_'+str(n_outs)
file_names = 'file_id_list_full.scp'


# configure trainer
cntk.device.try_set_default_device(cntk.gpu(0))
learning_rate = 0.008
num_epochs = 20
n_fold = 8
save_model_prefix = dir_base+'models/cntk_bisru_'
training_log_file = dir_base+'train_cntk_bisru.log'
rnn_model = rnn.CNTK_BiSRU(n_in=n_fold*n_ins, n_out=n_fold*n_outs, init_lr=learning_rate, momentum=0.95)
trainer = rnn_model.trainer


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

    # update learning rate if possible
    if epoch in [2, 4, 6, 8, 12, 16]:
        learning_rate *= 0.5
        rnn_model.learner.reset_learning_rate(cntk.learning_rate_schedule(learning_rate, cntk.UnitType.sample))

    # train
    train_error_list = []
    while not train_data_reader.is_finish():
        xx_train, yy_train = train_data_reader.load_one_partition()
        xx_train = rnn.re_order_batch(xx_train, fold=n_fold)
        yy_train = rnn.re_order_batch(yy_train, fold=n_fold)
        trainer.train_minibatch({rnn_model.input: xx_train, rnn_model.label: yy_train})
        curr_train_loss = trainer.previous_minibatch_loss_average
        train_error_list.append(curr_train_loss)
    train_data_reader.reset()

    # validation
    valid_loss_list = []
    while not valid_data_reader.is_finish():
        xx_valid, yy_valid = valid_data_reader.load_one_partition()
        xx_valid = rnn.re_order_batch(xx_valid, fold=n_fold)
        yy_valid = rnn.re_order_batch(yy_valid, fold=n_fold)
        curr_valid_loss = trainer.test_minibatch({rnn_model.input: xx_valid, rnn_model.label: yy_valid})
        valid_loss_list.append(curr_valid_loss)
    valid_data_reader.reset()

    # save model
    trainer.save_checkpoint(save_model_prefix + str(epoch))

    # calculate loss
    curr_valid_loss = np.mean(np.asarray(valid_loss_list)).item()
    curr_train_loss = np.mean(np.asarray(train_error_list)).item()

    curr_time = datetime.datetime.now().strftime('%Y-%m-%d %T')
    message = 'Epoch:%04d, Lr:%f, Train:%f, Valid:%f, Time:%f, Time:%s' \
              % (epoch, learning_rate, curr_train_loss, curr_valid_loss, time.time()-begin_time, curr_time)
    log_file.write(message+'\n')
    log_file.close()
    print message




















