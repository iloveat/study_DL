# -*- coding: utf-8 -*-
import cntk as C
import numpy as np
import time


def re_order_x(x, fold=4):
    num_frame = x.shape[0]
    x_dim = x.shape[1]
    if num_frame % fold == 0:
        n_add = 0
        x = x.reshape(-1, x_dim*fold)
    else:
        n_add = fold - num_frame % fold
        vec_add = np.zeros([n_add, x_dim], dtype=np.float32)
        x = np.concatenate((x, vec_add), axis=0).reshape(-1, x_dim*fold)
    return x, n_add, fold


def re_order_batch(xs, fold=4):
    new_xs = []
    for x in xs:
        new_x, n_add, fold = re_order_x(x, fold)
        new_xs.append(new_x)
    return new_xs


def loss_fun(output, label):
    length = C.sequence.reduce_sum(C.reduce_sum(output) * 0 + 1)
    return C.sequence.reduce_sum(C.reduce_sum(C.square(output - label))) / length


class CNTK_BiSRU(object):

    def __init__(self, n_in, n_out, init_lr, momentum):

        self.param1 = 512
        self.param2 = 256

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.input = C.sequence.input_variable(shape=(self.n_in,))
        self.label = C.sequence.input_variable(shape=(self.n_out,))

        self.three_dnn = C.layers.Sequential([
            C.layers.Dense(self.param1, activation=C.tanh, name='dnn_three_1'),
            C.layers.Dense(self.param1, activation=C.tanh, name='dnn_three_2'),
            C.layers.Dense(self.param1, activation=C.tanh, name='dnn_three_3')])
        self.final_dnn = C.layers.Dense(self.n_out, name='dnn_final')
        self.dnn_1 = C.layers.Dense(8 * self.param2, bias=False, name='dnn_1')
        self.dnn_2 = C.layers.Dense(8 * self.param2, bias=False, name='dnn_2')
        self.dnn_3 = C.layers.Dense(8 * self.param2, bias=False, name='dnn_3')
        self.dnn_4 = C.layers.Dense(8 * self.param2, bias=False, name='dnn_4')
        self.list_bias = []
        for i in xrange(16):
            self.list_bias.append(C.parameter(shape=(self.param2, ), name='bias_' + str(i)))

        self.output = self.model(self.input)

        self.loss = loss_fun(self.output, self.label)
        self.eval_err = loss_fun(self.output, self.label)

        self.lr_s = C.learning_rate_schedule(init_lr, C.UnitType.sample)
        self.mom_s = C.momentum_schedule(momentum)
        self.learner = C.momentum_sgd(self.output.parameters, lr=self.lr_s, momentum=self.mom_s)
        self.trainer = C.Trainer(self.output, (self.loss, self.eval_err), [self.learner])

    def bi_sru_layer(self, sru_1, index):
        f_1_f = C.sigmoid(sru_1[0 * self.param2 : 1 * self.param2] + self.list_bias[0 + index * 4])
        r_1_f = C.sigmoid(sru_1[1 * self.param2 : 2 * self.param2] + self.list_bias[1 + index * 4])
        c_1_f_r = (1 - f_1_f) * sru_1[2 * self.param2: 3 * self.param2]
        dec_c_1_f = C.layers.ForwardDeclaration('f_' + str(index))
        var_c_1_f = C.sequence.delay(dec_c_1_f, initial_state=0, time_step=1)
        nex_c_1_f = var_c_1_f * f_1_f + c_1_f_r
        dec_c_1_f.resolve_to(nex_c_1_f)
        h_1_f = r_1_f * C.tanh(nex_c_1_f) + (1 - r_1_f) * sru_1[3 * self.param2 : 4 * self.param2]

        f_1_b = C.sigmoid(sru_1[4 * self.param2 : 5 * self.param2] + self.list_bias[2 + index * 4])
        r_1_b = C.sigmoid(sru_1[5 * self.param2 : 6 * self.param2] + self.list_bias[3 + index * 4])
        c_1_b_r = (1 - f_1_b) * sru_1[6 * self.param2 : 7 * self.param2]
        dec_c_1_b = C.layers.ForwardDeclaration('b_' + str(index))
        var_c_1_b = C.sequence.delay(dec_c_1_b, time_step=-1)
        nex_c_1_b = var_c_1_b * f_1_b + c_1_b_r
        dec_c_1_b.resolve_to(nex_c_1_b)
        h_1_b = r_1_b * C.tanh(nex_c_1_b) + (1 - r_1_b) * sru_1[7 * self.param2 : 8 * self.param2]

        x = C.splice(h_1_f, h_1_b)
        return x

    def model(self, x):
        x = self.three_dnn(x)
        x = self.dnn_1(x)
        x = self.bi_sru_layer(x, 0)
        x = self.dnn_2(x)
        x = self.bi_sru_layer(x, 1)
        x = self.dnn_3(x)
        x = self.bi_sru_layer(x, 2)
        x = self.dnn_4(x)
        x = self.bi_sru_layer(x, 3)
        x = self.final_dnn(x)
        return x


def test_model():
    C.device.try_set_default_device(C.gpu(0))
    n_ins = 87
    n_outs = 187
    rnn_model = CNTK_BiSRU(n_in=n_ins, n_out=n_outs, init_lr=0.008, momentum=0.5)
    predictor = rnn_model.trainer.model
    t1 = time.time()
    output = predictor.eval([np.empty([822, 87], dtype=np.float32)])
    print output[0].shape
    print time.time()-t1


def test_fast_model(n_fold=8):
    num_frame = 822
    input_label = np.empty([num_frame, 87], dtype=np.float32)
    print input_label.shape

    # re-organise input vector
    if num_frame % n_fold == 0:
        n_add = 0
        input_label = input_label.reshape(-1, 87*n_fold)
    else:
        n_add = n_fold - num_frame % n_fold
        vec_add = np.zeros([n_add, 87], dtype=np.float32)
        input_label = np.concatenate((input_label, vec_add), axis=0).reshape(-1, 87*n_fold)

    C.device.try_set_default_device(C.gpu(0))
    n_ins = 87*n_fold
    n_outs = 187*n_fold
    rnn_model = CNTK_BiSRU(n_in=n_ins, n_out=n_outs, init_lr=0.008, momentum=0.5)
    predictor = rnn_model.trainer.model

    t1 = time.time()
    output = predictor.eval(input_label)
    print output[0].shape
    output = output[0].reshape(-1, 187)[:-n_add]
    print output.shape
    print time.time()-t1


# test_fast_model()
# test_fast_model()
# test_model()
# test_model()

















