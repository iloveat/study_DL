# -*- coding: utf-8 -*-
from mxnet.gluon import nn
from mxnet import nd


class BiSRU(nn.Block):
    def __init__(self, num_hidden, **kwargs):
        super(BiSRU, self).__init__(**kwargs)
        self.num_hidden = num_hidden    # 128
        with self.name_scope():
            self.dense1f = nn.Dense(self.num_hidden, use_bias=False)
            self.dense2f = nn.Dense(self.num_hidden, use_bias=False)
            self.dense3f = nn.Dense(self.num_hidden, use_bias=True, activation='sigmoid')
            self.dense4f = nn.Dense(self.num_hidden, use_bias=True, activation='sigmoid')
            self.dense1b = nn.Dense(self.num_hidden, use_bias=False)
            self.dense2b = nn.Dense(self.num_hidden, use_bias=False)
            self.dense3b = nn.Dense(self.num_hidden, use_bias=True, activation='sigmoid')
            self.dense4b = nn.Dense(self.num_hidden, use_bias=True, activation='sigmoid')

    def forward(self, x):           # (562, 256)
        xt_f_org = self.dense1f(x)  # (562, 128)
        xt_f_wav = self.dense2f(x)  # (562, 128)
        ft_f = self.dense3f(x)      # (562, 128)
        rt_f = self.dense4f(x)      # (562, 128)
        cr_f = (1-ft_f) * xt_f_wav  # (562, 128)
        hr_f = (1-rt_f) * xt_f_org  # (562, 128)
        ct_f = 0
        ht_f_list = []
        for i in range(x.shape[0]):
            ct_f = ft_f[i] * ct_f + cr_f[i]             # (128,)
            ht_f = rt_f[i] * nd.tanh(ct_f) + hr_f[i]    # (128,)
            ht_f_list.append(ht_f.reshape(shape=(-1, self.num_hidden)))

        hf = ht_f_list[0]
        for i in range(len(ht_f_list)-1):
            hf = nd.concat(hf, ht_f_list[i+1], dim=0)

        xt_b_org = self.dense1b(x)
        xt_b_wav = self.dense2b(x)
        ft_b = self.dense3b(x)
        rt_b = self.dense4b(x)
        cr_b = (1-ft_b) * xt_b_wav
        hr_b = (1-rt_b) * xt_b_org
        ct_b = 0
        ht_b_list = []
        for i in range(x.shape[0]):
            ct_b = ft_b[i] * ct_b + cr_b[i]
            ht_b = rt_b[i] * nd.tanh(ct_b) + hr_b[i]
            ht_b_list.append(ht_b.reshape(shape=(-1, self.num_hidden)))

        hb = ht_b_list[0]
        for i in range(len(ht_b_list)-1):
            hb = nd.concat(hb, ht_b_list[i+1], dim=0)

        out = nd.concat(hf, hb, dim=1)
        return out


class MXNET_BiSRU(nn.Block):
    def __init__(self, num_out=187, num_hidden=256, **kwargs):
        super(MXNET_BiSRU, self).__init__(**kwargs)
        assert num_hidden % 2 == 0
        self.num_out = num_out          # 187
        self.num_hidden = num_hidden    # 256
        with self.name_scope():
            net = self.net = nn.Sequential()
            net.add(nn.Dense(self.num_hidden, activation='tanh'))
            net.add(nn.Dense(self.num_hidden, activation='tanh'))
            net.add(nn.Dense(self.num_hidden, activation='tanh'))
            net.add(BiSRU(num_hidden=self.num_hidden/2))
            net.add(BiSRU(num_hidden=self.num_hidden/2))
            net.add(BiSRU(num_hidden=self.num_hidden/2))
            net.add(BiSRU(num_hidden=self.num_hidden/2))
            net.add(nn.Dense(self.num_out))

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        return out




