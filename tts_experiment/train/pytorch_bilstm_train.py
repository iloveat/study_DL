# -*- coding: utf-8 -*-
import numpy as np
import pytorch_bilstm as rnn
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import sys
sys.path.append('../util')
import path_provider as pap
import data_provider as dap


dir_base = '/home/brycezou/DATA/Audio/EM200/feature_16k/'
dir_lab_norm = 'nn_no_silence_lab_norm_87'
dir_cmp_norm = 'nn_norm_mgc_lf0_vuv_bap_187'
file_names = 'file_id_list_full.scp'
num_train_file = 200
num_valid_file = 60
n_ins = 87
n_outs = 187


# configure trainer
learning_rate = 0.008
num_epochs = 150
save_model_prefix = dir_base+'models/pytorch_bilstm_'
rnn_model = rnn.TORCH_BiLSTM(n_input=n_ins, n_output=n_outs)
rnn_model.cuda()
loss_function = nn.MSELoss()
optimizer = optim.SGD(rnn_model.parameters(), lr=learning_rate, momentum=0.5)


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


def data_iter(data_reader):
    while not data_reader.is_finish():
        train_set_x, train_set_y = data_reader.load_one_sample()
        yield autograd.Variable(torch.from_numpy(train_set_x).cuda()), autograd.Variable(torch.from_numpy(train_set_y).cuda())


for epoch in range(num_epochs):
    begin_time = time.time()
    log_file = open('pytorch_bilstm.log', 'a')

    # # train
    # train_error_list = []
    # while not train_data_reader.is_finish():
    #     xx_train, yy_train = train_data_reader.load_one_partition()
    #     trainer.train_minibatch({rnn_model.input: xx_train, rnn_model.label: yy_train})
    #     curr_train_loss = trainer.previous_minibatch_loss_average
    #     train_error_list.append(curr_train_loss)
    # train_data_reader.reset()

    # # validation
    # valid_loss_list = []
    # while not valid_data_reader.is_finish():
    #     xx_valid, yy_valid = valid_data_reader.load_one_partition()
    #     curr_valid_loss = trainer.test_minibatch({rnn_model.input: xx_valid, rnn_model.label: yy_valid})
    #     valid_loss_list.append(curr_valid_loss)
    # valid_data_reader.reset()

    # # save model
    # trainer.save_checkpoint(save_model_prefix + str(epoch))

    train_error_list = []
    for xx_train, yy_train in data_iter(train_data_reader):
        rnn_model.zero_grad()
        rnn_model.hidden = rnn_model.init_hidden()
        output = rnn_model(xx_train)
        loss = loss_function(output, yy_train)
        loss.backward()
        optimizer.step()
        train_error_list.append(loss)
    train_data_reader.reset()

    valid_loss_list = []
    with torch.no_grad():
        for xx_valid, yy_valid in data_iter(valid_data_reader):
            output = rnn_model(xx_valid)
            loss = loss_function(output, yy_valid)
            valid_loss_list.append(loss)
    valid_data_reader.reset()

    torch.save(rnn_model, save_model_prefix + str(epoch))

    # calculate loss
    curr_valid_loss = np.mean(np.asarray(valid_loss_list)).item()
    curr_train_loss = np.mean(np.asarray(train_error_list)).item()

    # update learning rate if possible
    # learning_rate = 0.008
    # rnn_model.learner.reset_learning_rate(cntk.learning_rate_schedule(learning_rate, cntk.UnitType.sample))

    message = 'Epoch:%04d, Lr:%f, Train:%f, Valid:%f, Time:%f' \
              % (epoch, learning_rate, curr_train_loss, curr_valid_loss, time.time()-begin_time)
    log_file.write(message+'\n')
    log_file.close()
    print message




















