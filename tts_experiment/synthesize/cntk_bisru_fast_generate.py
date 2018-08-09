# -*- coding: utf-8 -*-
import numpy as np
import os
import cntk
import time
import sys
sys.path.append('../train')
import cntk_bisru_fast as rnn
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


saved_epoch = 18
saved_ckp_model = dir_base+'models/cntk_bisru_'+str(saved_epoch)
output_dir = dir_base+'synthesis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


file_list = pap.read_file_list(dir_base+file_names)
lab_norm_file_list = pap.prepare_file_path_list(file_list, dir_base+dir_lab_norm, '.lab')
generate_file_list = pap.prepare_file_path_list(file_list, output_dir, '.cmp')


n_fold = 8
cntk.device.try_set_default_device(cntk.gpu(0))
rnn_model = rnn.CNTK_BiSRU(n_in=n_fold*n_ins, n_out=n_fold*n_outs, init_lr=0.008, momentum=0.95)
rnn_model.trainer.restore_from_checkpoint(saved_ckp_model)
predictor = rnn_model.trainer.model


# generate cmp file
num_file = len(lab_norm_file_list)
for i in range(num_file):
    t1 = time.time()
    # read label feature from file
    features = np.fromfile(lab_norm_file_list[i], dtype=np.float32)
    # evaluation
    features = features[:(n_ins*(features.size/n_ins))]
    input_labels = features.reshape((-1, n_ins))

    input_labels, n_add, fold = rnn.re_order_x(input_labels, fold=n_fold)
    predicted_parameter = predictor.eval([input_labels]*1)[0]
    predicted_parameter = np.array(predicted_parameter, dtype=np.float32).reshape(-1, n_outs)
    if n_add != 0:
        predicted_parameter = predicted_parameter[:-n_add]

    # write to cmp file
    predicted_parameter.tofile(generate_file_list[i])
    print '%04d of %04d predicted: %s' % (i+1, num_file, lab_norm_file_list[i])
    print time.time()-t1


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























