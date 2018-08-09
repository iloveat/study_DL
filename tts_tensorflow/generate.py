# -*- coding: utf-8 -*-
import data_provider as dp
from mlpg_fast import MLParameterGenerationFast as MLParameterGeneration
import numpy as np
import os
import errno
from generate_wave import generate_wave
import tensorflow as tf


def read_file_list(file_name):
    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()
    return file_lists


def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)
    return file_name_list


def feature_normalisation(in_file_list, out_file_list, mean_vector, std_vector, feature_dimension):
    file_number = len(in_file_list)
    try:
        assert len(in_file_list) == len(out_file_list)
    except AssertionError:
        print 'The input and output file numbers do not match! %d vs %d' % (len(in_file_list), len(out_file_list))
        raise
    try:
        assert mean_vector.size == feature_dimension and std_vector.size == feature_dimension
    except AssertionError:
        print 'The dimensionality of the mean and standard derivation vectors do not match'
        raise
    for j in range(file_number):
        org_feature, curr_frame_number = dp.load_binary_file_frame(in_file_list[j], feature_dimension)
        mean_matrix = np.tile(mean_vector, (curr_frame_number, 1))
        std_matrix = np.tile(std_vector, (curr_frame_number, 1))
        norm_feature = org_feature * std_matrix + mean_matrix
        dp.array_to_binary_file(norm_feature, out_file_list[j])
    pass


def acoustic_decomposition(in_file_list, dimension_all):
    param_dim_dict = {'mgc': 180, 'vuv': 1, 'lf0': 3, 'bap': 3}
    file_ext_dict = {'mgc': '.mgc', 'lf0': '.lf0', 'bap': '.bap'}

    # load parameter variance
    param_var_dir = \
        '/home/brycezou/works/tuling_tts_train/egs/005_hdl_test/s1/experiments/ht-500-data/acoustic_model/data/var'
    var_file_dict = {}
    for feature_name in param_dim_dict.keys():
        var_file_dict[feature_name] = os.path.join(param_var_dir, feature_name + '_' + str(param_dim_dict[feature_name]))
    var_dict = {}
    for feature_name in var_file_dict.keys():
        var_value, _ = dp.load_binary_file_frame(var_file_dict[feature_name], 1)
        var_value = np.reshape(var_value, (param_dim_dict[feature_name], 1))
        var_dict[feature_name] = var_value

    # parameter start index
    dimension_index = 0
    stream_start_index = {}
    for feature_name in param_dim_dict.keys():
        stream_start_index[feature_name] = dimension_index
        dimension_index += param_dim_dict[feature_name]

    wave_feature_types = ['mgc', 'lf0', 'bap']
    inf_float = -1.0e+10

    mlpg = MLParameterGeneration()

    # one cmp file per loop
    for file_name in in_file_list:
        dir_name = os.path.dirname(file_name)
        file_id = os.path.splitext(os.path.basename(file_name))[0]

        # load cmp data from file
        features_all, frame_number = dp.load_binary_file_frame(file_name, dimension_all)

        # one type of features per loop
        for feature_name in wave_feature_types:
            curr_feature = features_all[:, stream_start_index[feature_name]: \
                                        stream_start_index[feature_name]+param_dim_dict[feature_name]]
            var = var_dict[feature_name]
            var = np.transpose(np.tile(var, frame_number))
            gen_features = mlpg.generation(curr_feature, var, param_dim_dict[feature_name]/3)

            if feature_name in ['lf0', 'F0']:
                if stream_start_index.has_key('vuv'):
                    vuv_feature = features_all[:, stream_start_index['vuv']:stream_start_index['vuv']+1]
                    for i in xrange(frame_number):
                        if vuv_feature[i, 0] < 0.5:
                            gen_features[i, 0] = inf_float

            new_file_name = os.path.join(dir_name, file_id + file_ext_dict[feature_name])
            dp.array_to_binary_file(gen_features, new_file_name)
            print 'wrote to file %s' % new_file_name
    pass


dir_base = '/home/brycezou/works/tuling_tts_train/egs/005_hdl_test/s1/experiments/ht-500-data/acoustic_model/data/'
dir_lab_norm = 'nn_no_silence_lab_norm_87'
file_ids = 'synthesis/test_id_list.scp'
save_dir = 'synthesis'


try:
    os.makedirs(save_dir)
except OSError as e:
    if e.errno == errno.EEXIST:
        pass
    else:
        print 'Failed to create generation directory %s' % save_dir
        raise


file_list = read_file_list(file_ids)
lab_norm_file_list = prepare_file_path_list(file_list, dir_base+dir_lab_norm, '.lab')


n_ins = 87
n_outs = 187
generate_file_list = prepare_file_path_list(file_list, save_dir, '.cmp')


saved_checkpoint_model = '../model/tts/tf-sru/tf-sru-0198'
sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph(saved_checkpoint_model+'.meta')
saver.restore(sess, saved_checkpoint_model)

graph = tf.get_default_graph()
input_X = graph.get_tensor_by_name('input_x:0')
seq_len = graph.get_tensor_by_name('seq_len:0')
predict_Y = graph.get_tensor_by_name('predict_y:0')


# generate cmp file
num_file = len(lab_norm_file_list)
for i in range(num_file):
    # read label feature from file
    fid_lab = open(lab_norm_file_list[i], 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    # evaluation
    features = features[:(n_ins*(features.size/n_ins))]
    input_labels = features.reshape((1, -1, n_ins))

    # predicted_parameter = predict_Y.eval(feed_dict={input_X: input_labels, seq_len: [input_labels.shape[1]]})
    predicted_parameter = sess.run(predict_Y, feed_dict={input_X: input_labels, seq_len: [input_labels.shape[1]]})

    # write to cmp file
    fid = open(generate_file_list[i], 'wb')
    predicted_parameter.tofile(fid)
    fid.close()
    print '%04d of %04d generated: %s' % (i+1, num_file, lab_norm_file_list[i])


normalization_file = 'norm_info_mgc_lf0_vuv_bap_187_MVN.dat'
fid = open(dir_base+normalization_file, 'rb')
cmp_mean_std = np.fromfile(fid, dtype=np.float32)
fid.close()
cmp_mean_std = cmp_mean_std.reshape((2, -1))
cmp_mean_vector = cmp_mean_std[0, ]
cmp_std_vector = cmp_mean_std[1, ]


feature_normalisation(in_file_list=generate_file_list, out_file_list=generate_file_list,
                      mean_vector=cmp_mean_vector, std_vector=cmp_std_vector, feature_dimension=n_outs)


acoustic_decomposition(in_file_list=generate_file_list, dimension_all=n_outs)

output_dir = '/home/brycezou/software/turorials/gluon-tutorials/mxnet_study/tts_tensorflow/synthesis'
generate_wave(output_dir=output_dir, file_id_list=file_list)








