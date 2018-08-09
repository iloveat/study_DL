# -*- coding: utf-8 -*-
import numpy as np
import os
import mxnet as mx
from mxnet import init
from mxnet import nd
import sys
sys.path.append('../train')
import mxnet_bisru as rnn
sys.path.append('../util')
import path_provider as pap
import data_normalization as dan
import world_vocoder as wov
import acoustic_decomposition as acd


dir_base = '/home/brycezou/DATA/Audio/EM2000/feature_16k/'
b_16k = True


if b_16k:
    n_ins = 87
    n_outs = 187
else:
    n_ins = 87
    n_outs = 193
dir_lab_norm = 'nn_no_silence_lab_norm_'+str(n_ins)
file_names = 'test_id_list.scp'


saved_epoch = 0
saved_ckp_model = dir_base+'models/mxnet_bisru_' + '%d.params' % saved_epoch
output_dir = dir_base+'synthesis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


file_list = pap.read_file_list(dir_base+file_names)
lab_norm_file_list = pap.prepare_file_path_list(file_list, dir_base+dir_lab_norm, '.lab')
generate_file_list = pap.prepare_file_path_list(file_list, output_dir, '.cmp')


ctx = mx.gpu()
net = rnn.MXNET_BiSRU(num_out=n_outs, num_hidden=512)
net.initialize(ctx=ctx, init=init.Xavier())
net.hybridize()
net.load_params(saved_ckp_model, ctx=ctx)
predictor = net


# generate cmp file
num_file = len(lab_norm_file_list)
for i in range(num_file):
    # read label feature from file
    features = np.fromfile(lab_norm_file_list[i], dtype=np.float32)
    # evaluation
    features = features[:(n_ins*(features.size/n_ins))]
    input_labels = features.reshape((-1, n_ins))
    predicted_parameter = predictor(nd.array(input_labels, ctx=ctx))
    predicted_parameter = predicted_parameter.asnumpy()
    # write to cmp file
    predicted_parameter = np.array(predicted_parameter, dtype=np.float32)
    predicted_parameter.tofile(generate_file_list[i])
    print '%04d of %04d predicted: %s' % (i+1, num_file, lab_norm_file_list[i])


cmp_norm_file = 'norm_info_mgc_lf0_vuv_bap_%d_MVN.dat' % n_outs
cmp_mean_std = np.fromfile(dir_base+cmp_norm_file, dtype=np.float32)
cmp_mean_std = cmp_mean_std.reshape((2, -1))
cmp_mean_vector = cmp_mean_std[0, ]
cmp_std_vector = cmp_mean_std[1, ]


dan.feature_normalisation(in_file_list=generate_file_list, out_file_list=generate_file_list,
                          mean_vector=cmp_mean_vector, std_vector=cmp_std_vector, feature_dimension=n_outs)


cmp_vars_dir = dir_base+'var'
acd.acoustic_decomposition(in_file_list=generate_file_list, dimension_all=n_outs, param_var_dir=cmp_vars_dir,
                           b_16k=b_16k)


wov.generate_wave(output_dir=output_dir, file_id_list=file_list, b_16k=b_16k)























