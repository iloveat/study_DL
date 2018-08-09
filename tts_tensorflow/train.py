# -*- coding: utf-8 -*-
import tensorflow as tf
import data_provider as dp
import os
import sru_cell as sru
import time


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


def data_iter(data_reader):
    while not data_reader.is_finish():
        train_set_x, train_set_y = data_reader.load_one_sample()
        yield train_set_x, train_set_y


dir_base = '/home/brycezou/works/tuling_tts_train/egs/005_hdl_test/s1/experiments/ht-500-data/acoustic_model/data/'
dir_lab_norm = 'nn_no_silence_lab_norm_87'
dir_cmp_norm = 'nn_norm_mgc_lf0_vuv_bap_187'
file_ids = 'file_id_list_full.scp'


file_list = read_file_list(dir_base+file_ids)
cmp_norm_file_list = prepare_file_path_list(file_list, dir_base+dir_cmp_norm, '.cmp')
lab_norm_file_list = prepare_file_path_list(file_list, dir_base+dir_lab_norm, '.lab')


num_train_file = 222
num_valid_file = 37
train_x_file_list = lab_norm_file_list[0:num_train_file]
train_y_file_list = cmp_norm_file_list[0:num_train_file]
valid_x_file_list = lab_norm_file_list[num_train_file:num_train_file+num_valid_file]
valid_y_file_list = cmp_norm_file_list[num_train_file:num_train_file+num_valid_file]


n_ins = 87
n_outs = 187
buffer_size = 20000


train_data_reader = dp.ListDataProvider(x_file_list=train_x_file_list, y_file_list=train_y_file_list, n_ins=n_ins,
                                        n_outs=n_outs, buffer_size=buffer_size, shuffle=True)
valid_data_reader = dp.ListDataProvider(x_file_list=valid_x_file_list, y_file_list=valid_y_file_list, n_ins=n_ins,
                                        n_outs=n_outs, buffer_size=buffer_size, shuffle=False)
train_data_reader.reset()
valid_data_reader.reset()


dim_feature = 87
num_output = 187

learning_rate = 0.002
num_hidden = 128
num_epoch = 200
batch_size = 1


def create_model(inputs, sequence_length, name):
    """
    we assume that batch size = 1
    for example:
        if inputs shape is (1, 562, 87) <= (batch size, frame number, feature dim),
        then sequence_length is [562], which implies frame number of current wave
    """
    x = tf.reshape(inputs, [-1, dim_feature])  # (562, 87)

    w1 = tf.Variable(tf.random_normal([dim_feature, num_hidden]))  # (87, 256)
    b1 = tf.Variable(tf.random_normal([num_hidden], mean=1.0))  # (256,)
    y1 = tf.nn.tanh(tf.matmul(x, w1) + b1)  # (562, 256)

    w2 = tf.Variable(tf.random_normal([num_hidden, num_hidden]))  # (256, 256)
    b2 = tf.Variable(tf.random_normal([num_hidden], mean=1.0))  # (256,)
    y2 = tf.nn.tanh(tf.matmul(y1, w2) + b2)  # (562, 256)

    w3 = tf.Variable(tf.random_normal([num_hidden, num_hidden]))  # (256, 256)
    b3 = tf.Variable(tf.random_normal([num_hidden], mean=1.0))  # (256,)
    y3 = tf.nn.tanh(tf.matmul(y2, w3) + b3)  # (562, 256)

    y4 = tf.reshape(y3, [1, sequence_length[0], num_hidden])  # (1, 562, 256)

    # Stack 2 lstm layers
    # layers = [tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) for _ in range(2)]
    # layers = [tf.contrib.rnn.GRUCell(num_hidden) for _ in range(2)]
    layers = [sru.SRUCell(num_hidden, True) for _ in range(2)]
    multi_layer_rnn = tf.contrib.rnn.MultiRNNCell(layers)

    # outputs shape: (1, 562, 256)
    outputs, _ = tf.nn.dynamic_rnn(multi_layer_rnn, y4, sequence_length=sequence_length, dtype=tf.float32)
    w0 = tf.Variable(tf.random_normal([num_hidden, num_output]))  # (256, 187)
    b0 = tf.Variable(tf.random_normal([num_output]))  # (187,)

    return tf.add(tf.matmul(tf.reshape(outputs, [-1, num_hidden]), w0), b0, name=name)  # (562, 187)


tf.reset_default_graph()

# [batch size, time step, feature dims]
input_X = tf.placeholder(tf.float32, [1, None, dim_feature], name="input_x")
input_Y = tf.placeholder(tf.float32, [None, num_output])
seq_len = tf.placeholder(tf.int32, [None, ], name="seq_len")


predict_Y = create_model(input_X, seq_len, name="predict_y")


# l2 = sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = tf.reduce_mean(tf.nn.l2_loss(predict_Y-input_Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


fine_tune = False
saved_checkpoint_model = '../model/tts/tf-sru/tf-sru-0024'


saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


if fine_tune:
    saver.restore(sess, saved_checkpoint_model)


for i in range(num_epoch):
    train_data_reader.reset()
    train_loss = 0.0
    for data, label in data_iter(train_data_reader):
        data = data.reshape((1, -1, 87))
        label = label.reshape((-1, 187))
        sess.run(optimizer, feed_dict={input_X: data, input_Y: label, seq_len: [data.shape[1]]})
        train_loss += sess.run(loss, feed_dict={input_X: data, input_Y: label, seq_len: [data.shape[1]]})
    saver.save(sess, '../model/tts/tf-sru/tf-sru-%04d' % i)
    print 'epoch: %4d, train_loss: %f' % (i, train_loss)








