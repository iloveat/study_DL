# -*- coding: utf-8 -*-
import cntk as C
from cntk.layers import Sequential, Recurrence, LSTM, Dense
import numpy as np


def loss_fun(output, label):
    length = C.sequence.reduce_sum(C.reduce_sum(output) * 0 + 1)
    return C.sequence.reduce_sum(C.reduce_sum(C.square(output - label))) / length


class CNTK_BiLSTM(object):

    def __init__(self, n_in, n_out, init_lr, momentum):

        self.param1 = 512
        self.param2 = 256

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.input = C.sequence.input_variable(shape=(self.n_in,))
        self.label = C.sequence.input_variable(shape=(self.n_out,))

        self.three_dnn = Sequential([Dense(self.param1, activation=C.tanh),
                                     Dense(self.param1, activation=C.tanh),
                                     Dense(self.param1, activation=C.tanh)])
        self.rnn_layer1 = Sequential([(Recurrence(LSTM(self.param2)), Recurrence(LSTM(self.param2), go_backwards=True)),
                                      C.splice])
        self.rnn_layer2 = Sequential([(Recurrence(LSTM(self.param2)), Recurrence(LSTM(self.param2), go_backwards=True)),
                                      C.splice])
        self.final_dnn = Dense(self.n_out)

        self.output = self.model(self.input)

        self.loss = loss_fun(self.output, self.label)
        self.eval_err = loss_fun(self.output, self.label)

        self.lr_s = C.learning_rate_schedule(init_lr, C.UnitType.sample)
        self.mom_s = C.momentum_schedule(momentum)
        self.learner = C.momentum_sgd(self.output.parameters, lr=self.lr_s, momentum=self.mom_s)
        self.trainer = C.Trainer(self.output, (self.loss, self.eval_err), [self.learner])

    def model(self, x):
        x = self.three_dnn(x)
        x = self.rnn_layer1(x)
        x = self.rnn_layer2(x)
        x = self.final_dnn(x)
        return x


def test_model():
    C.device.try_set_default_device(C.gpu(0))
    n_ins = 87
    n_outs = 187
    rnn_model = CNTK_BiLSTM(n_in=n_ins, n_out=n_outs, init_lr=0.008, momentum=0.5)
    predictor = rnn_model.trainer.model
    output = predictor.eval([np.empty([622, 87], dtype=np.float32)])
    print output[0].shape


# test_model()






























