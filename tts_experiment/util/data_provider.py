# -*- coding: utf-8 -*-
import numpy as np
import random


def load_binary_file_frame(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0, 'specified dimension %s not compatible with data' % dimension
    frame_number = features.size / dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return features, frame_number


def array_to_binary_file(data, output_file_name):
    data = np.array(data, dtype=np.float32)
    fid = open(output_file_name, 'wb')
    data.tofile(fid)
    fid.close()


class ListDataProvider(object):
    """
    This class provides an interface to load data into CPU/GPU memory utterance by utterance or block by block.
    In speech synthesis, usually we are not able to load all the training data/evaluation data into RAMs,
    we will do the following three steps:
    - Step 1: a data provide will load part of the data into a buffer
    - Step 2: training a DNN by using the data from the buffer
    - Step 3: Iterate step 1 and 2 until all the data are used for DNN training.
    Until now, one epoch of DNN training is finished.
    The utterance-by-utterance data loading will be useful when sequential training is used,
    while block-by-block loading will be used when the order of frames is not important.
    This provide assumes binary format with float32 precision without any header (e.g. HTK header).
    """
    def __init__(self, x_file_list, y_file_list, n_ins=0, n_outs=0, shuffle=False):
        """Initialise a data provider
        :param x_file_list: list of file names for the input files to DNN
        :type x_file_list: python list
        :param y_file_list: list of files for the output files to DNN
        :param n_ins: the dimensionality for input feature
        :param n_outs: the dimensionality for output features
        :param shuffle: True/False. To indicate whether the file list will be shuffled. When loading data block by block, the data in the buffer will be shuffle no matter this value is True or False.
        """
        self.n_ins = n_ins
        self.n_outs = n_outs

        try:
            assert len(x_file_list) > 0
            assert len(y_file_list) > 0
            assert len(x_file_list) == len(y_file_list)
        except AssertionError:
            print 'x_file_list and y_file_list size exception'
            raise

        self.x_files_list = x_file_list
        self.y_files_list = y_file_list

        if shuffle:
            random.seed(271638)
            random.shuffle(self.x_files_list)
            random.seed(271638)
            random.shuffle(self.y_files_list)

        self.list_size = len(self.x_files_list)
        self.file_index = 0
        self.end_reading = False

    def __iter__(self):
        return self

    def reset(self):
        """
        When all the files in the file list have been used for DNN training,
        reset the data provider to start a new epoch.
        """
        self.file_index = 0
        self.end_reading = False

    def is_finish(self):
        return self.end_reading

    def load_one_partition(self):
        temp_set_x, temp_set_y = self.load_next_utterance()
        return temp_set_x, temp_set_y

    def load_next_utterance(self):
        """
        Load the data for one utterance.
        This function will be called when utterance-by-utterance loading is required (e.g., sequential training).
        """
        temp_x = []
        temp_y = []

        batch_size = 4

        for i in xrange(batch_size):

            in_features, lab_frame_number = load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
            out_features, out_frame_number = load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)

            frame_number = lab_frame_number
            # we allow small difference here. may not be correct, but sometimes, there is one/two frames difference
            if abs(lab_frame_number - out_frame_number) < 5:
                if lab_frame_number > out_frame_number:
                    frame_number = out_frame_number
            else:
                base_file_name = self.x_files_list[self.file_index].split('/')[-1].split('.')[0]
                print "the number of frames in label and acoustic features are different: %d vs %d (%s)" % \
                      (lab_frame_number, out_frame_number, base_file_name)
                raise

            temp_set_y = out_features[0:frame_number, ]
            temp_set_x = in_features[0:frame_number, ]

            temp_x.append(np.asarray(temp_set_x, dtype=np.float32))
            temp_y.append(np.asarray(temp_set_y, dtype=np.float32))

            self.file_index += 1

        if self.file_index + batch_size-1 >= self.list_size:
            self.file_index = 0
            self.end_reading = True

        return temp_x, temp_y

    def load_one_sample(self):
        lab_features, lab_frame_number = load_binary_file_frame(self.x_files_list[self.file_index], self.n_ins)
        cmp_features, cmp_frame_number = load_binary_file_frame(self.y_files_list[self.file_index], self.n_outs)

        frame_number = lab_frame_number
        # we allow small difference here. may not be correct, but sometimes, there is one/two frames difference
        if abs(lab_frame_number - cmp_frame_number) < 5:
            if lab_frame_number > cmp_frame_number:
                frame_number = cmp_frame_number
        else:
            base_file_name = self.x_files_list[self.file_index].split('/')[-1].split('.')[0]
            print "the number of frames in label and acoustic features are different: %d vs %d (%s)" % \
                  (lab_frame_number, cmp_frame_number, base_file_name)
            raise

        self.file_index += 1
        if self.file_index >= self.list_size:
            self.file_index = 0
            self.end_reading = True

        temp_x = np.asarray(lab_features[0:frame_number, ], dtype=np.float32)
        temp_y = np.asarray(cmp_features[0:frame_number, ], dtype=np.float32)
        return temp_x, temp_y
























