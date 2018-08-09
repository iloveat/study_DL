# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import time


class TORCH_BiLSTM(nn.Module):

    def __init__(self, n_input, n_output, num_layer=2, bidirectional=False):
        super(TORCH_BiLSTM, self).__init__()
        self.dim_input = n_input
        self.dim_output = n_output
        self.dim_dense_layer = 512
        self.dim_hidden_rnn = 200
        self.num_layers_rnn = num_layer
        self.bidirectional = bidirectional
        self.dim_h0_c0 = self.num_layers_rnn*2 if self.bidirectional else self.num_layers_rnn
        self.dim_last_dense = self.dim_hidden_rnn*2 if self.bidirectional else self.dim_hidden_rnn

        self.dense1 = nn.Linear(self.dim_input, self.dim_dense_layer)
        self.dense2 = nn.Linear(self.dim_dense_layer, self.dim_dense_layer)
        self.dense3 = nn.Linear(self.dim_dense_layer, self.dim_dense_layer)
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(self.dim_dense_layer, self.dim_hidden_rnn,
                            num_layers=self.num_layers_rnn, bidirectional=self.bidirectional)
        self.dense4 = nn.Linear(self.dim_last_dense, self.dim_output)

    def init_hidden(self):
        h0 = autograd.Variable(torch.zeros(self.dim_h0_c0, 1, self.dim_hidden_rnn))
        c0 = autograd.Variable(torch.zeros(self.dim_h0_c0, 1, self.dim_hidden_rnn))
        return h0, c0

    def forward(self, x):  # 622x87
        x = F.tanh(self.dense1(x))  # 622x512
        x = F.tanh(self.dense2(x))  # 622x512
        x = F.tanh(self.dense3(x))  # 622x512
        y = x.view(len(x), 1, -1)   # 622x1x512
        x, self.hidden = self.lstm(y, self.hidden)  # 622x1x200, 2x1x200 | 622x1x400, 4x1x200
        x = self.dense4(x.view(len(x), -1))  # 622x187
        return x


def test_model():
    model = TORCH_BiLSTM(87, 187)
    model.hidden = model.init_hidden()
    model.cuda()

    xx = torch.cat([autograd.Variable(torch.ones(1, 87).cuda()) for i in range(622)])
    print xx
    t1 = time.time()
    yy = model(xx)
    print time.time()-t1
    print yy


test_model()








