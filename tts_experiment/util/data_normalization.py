import numpy as np
import data_provider as dap


def feature_normalisation(in_file_list, out_file_list, mean_vector, std_vector, feature_dimension):
    assert mean_vector.size == feature_dimension and std_vector.size == feature_dimension
    for j in range(len(in_file_list)):
        org_feature, curr_frame_number = dap.load_binary_file_frame(in_file_list[j], feature_dimension)
        mean_matrix = np.tile(mean_vector, (curr_frame_number, 1))
        std_matrix = np.tile(std_vector, (curr_frame_number, 1))
        norm_feature = org_feature * std_matrix + mean_matrix
        dap.array_to_binary_file(norm_feature, out_file_list[j])
























