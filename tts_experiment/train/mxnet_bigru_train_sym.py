import logging
import mxnet as mx
from mxnet import nd
import sys
sys.path.append('../util')
import path_provider as pap
import data_provider as dap


dir_base = '/home/brycezou/DATA/Audio/EM2000/feature_16k/'
b_16k = True
num_train_file = 1600
num_valid_file = 399


if b_16k:
    n_ins = 87
    n_outs = 187
else:
    n_ins = 87
    n_outs = 193
dir_lab_norm = 'nn_no_silence_lab_norm_'+str(n_ins)
dir_cmp_norm = 'nn_norm_mgc_lf0_vuv_bap_'+str(n_outs)
file_names = 'file_id_list_full.scp'


# prepare the training data
file_list = pap.read_file_list(dir_base+file_names)
cmp_norm_file_list = pap.prepare_file_path_list(file_list, dir_base+dir_cmp_norm, '.cmp')
lab_norm_file_list = pap.prepare_file_path_list(file_list, dir_base+dir_lab_norm, '.lab')

train_x_file_list = lab_norm_file_list[0:num_train_file]
train_y_file_list = cmp_norm_file_list[0:num_train_file]
valid_x_file_list = lab_norm_file_list[num_train_file:num_train_file+num_valid_file]
valid_y_file_list = cmp_norm_file_list[num_train_file:num_train_file+num_valid_file]

train_data_reader = dap.ListDataProvider(x_file_list=train_x_file_list, y_file_list=train_y_file_list,
                                         n_ins=n_ins, n_outs=n_outs, shuffle=True)
valid_data_reader = dap.ListDataProvider(x_file_list=valid_x_file_list, y_file_list=valid_y_file_list,
                                         n_ins=n_ins, n_outs=n_outs, shuffle=False)
train_data_reader.reset()
valid_data_reader.reset()


class SimpleIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, label_names, label_shapes, data_reader):
        super(SimpleIter, self).__init__()
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self._data_reader = data_reader

    def __iter__(self):
        return self

    def reset(self):
        self._data_reader.reset()

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if not self._data_reader.is_finish():
            train_set_x, train_set_y = self._data_reader.load_one_sample()
            data_, label_ = nd.array(train_set_x), nd.array(train_set_y)
            return mx.io.DataBatch([data_], [label_])
        else:
            raise StopIteration


def symbol_net():
    param1 = 512
    param2 = 256
    net_sym = mx.sym.Variable('data')
    net_sym = mx.sym.FullyConnected(data=net_sym, num_hidden=param1, name='fc1')
    net_sym = mx.sym.Activation(data=net_sym, act_type='tanh', name='tanh1')
    net_sym = mx.sym.FullyConnected(data=net_sym, num_hidden=param1, name='fc2')
    net_sym = mx.sym.Activation(data=net_sym, act_type='tanh', name='tanh2')
    net_sym = mx.sym.FullyConnected(data=net_sym, num_hidden=param1, name='fc3')
    net_sym = mx.sym.Activation(data=net_sym, act_type='tanh', name='tanh3')

    # RNN cell takes input of shape (time, batch, feature)
    # the RNN cell output is of shape (time, batch, dim)
    net_sym = mx.sym.Reshape(data=net_sym, shape=(-1, 1, param1))
    net_sym = mx.sym.RNN(data=net_sym, state_size=param2, num_layers=2, mode='gru', name='GRU', bidirectional=True,
                         parameters=mx.sym.Variable('GRU_bias'), state=mx.sym.Variable('GRU_weight'))
    net_sym = mx.sym.Reshape(data=net_sym, shape=(-1, param1))
    net_sym = mx.sym.FullyConnected(data=net_sym, num_hidden=n_outs, name='fc4')
    input_label = mx.sym.Variable('input_label')
    prd = mx.sym.LinearRegressionOutput(data=net_sym, label=input_label, name='regression')
    # mx.viz.plot_network(prd, shape={'data': (562, 87)}).view()
    return prd


learning_rate = 0.002
batch_size = 1


train_iter = SimpleIter(['data'], [(batch_size, n_ins)], ['input_label'], [(batch_size, n_outs)], train_data_reader)
valid_iter = SimpleIter(['data'], [(batch_size, n_ins)], ['input_label'], [(batch_size, n_outs)], valid_data_reader)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')
mod = mx.mod.Module(symbol=symbol_net(),
                    context=mx.gpu(0),
                    data_names=['data'],
                    label_names=['input_label'])

save_model_prefix = dir_base+'models/mxnet_bigru_sym_'

mod.fit(train_data=train_iter,
        eval_data=valid_iter,
        num_epoch=100,
        eval_metric='mse',
        batch_end_callback=mx.callback.Speedometer(batch_size, 1),
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        optimizer='sgd',
        optimizer_params={'learning_rate': learning_rate, 'momentum': 0.5, 'wd': 5e-4},
        epoch_end_callback=mx.callback.do_checkpoint(save_model_prefix))




































