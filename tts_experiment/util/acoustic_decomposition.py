from mlpg_fast import MLParameterGenerationFast as MLParameterGeneration
import data_provider as dap
import numpy as np
import os


def acoustic_decomposition(in_file_list, dimension_all, param_var_dir, b_16k=True):
    if b_16k:  # 16k
        param_dim_dict = {'mgc': 180, 'vuv': 1, 'lf0': 3, 'bap': 3}
    else:  # 24k
        param_dim_dict = {'mgc': 180, 'vuv': 1, 'lf0': 3, 'bap': 9}
    file_ext_dict = {'mgc': '.mgc', 'lf0': '.lf0', 'bap': '.bap'}

    # load parameter variance
    var_file_dict = {}
    for f_name in param_dim_dict.keys():
        var_file_dict[f_name] = os.path.join(param_var_dir, f_name + '_' + str(param_dim_dict[f_name]))
    var_dict = {}
    for f_name in var_file_dict.keys():
        var_value, _ = dap.load_binary_file_frame(var_file_dict[f_name], 1)
        var_value = np.reshape(var_value, (param_dim_dict[f_name], 1))
        var_dict[f_name] = var_value

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
        features_all, frame_number = dap.load_binary_file_frame(file_name, dimension_all)

        # one type of features per loop
        for feature_name in wave_feature_types:
            curr_feature = features_all[:, stream_start_index[feature_name]: stream_start_index[feature_name]+param_dim_dict[feature_name]]
            var = var_dict[feature_name]
            var = np.transpose(np.tile(var, frame_number))
            gen_features = mlpg.generation(curr_feature, var, param_dim_dict[feature_name]/3)

            if feature_name in ['lf0', 'F0']:
                if 'vuv' in stream_start_index:
                    vuv_feature = features_all[:, stream_start_index['vuv']:stream_start_index['vuv']+1]
                    for i in xrange(frame_number):
                        if vuv_feature[i, 0] < 0.5:
                            gen_features[i, 0] = inf_float

            new_file_name = os.path.join(dir_name, file_id + file_ext_dict[feature_name])
            dap.array_to_binary_file(gen_features, new_file_name)
            print 'wrote to file %s' % new_file_name
    pass























