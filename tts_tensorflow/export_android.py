# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


saved_checkpoint_model = '../model/tts/tf-sru/tf-sru-0198'
sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph(saved_checkpoint_model+'.meta')
saver.restore(sess, saved_checkpoint_model)

tf.train.write_graph(sess.graph_def, '.', 'tfdroid.pbtxt')


graph = tf.get_default_graph()
input_X = graph.get_tensor_by_name('input_x:0')
seq_len = graph.get_tensor_by_name('seq_len:0')
predict_Y = graph.get_tensor_by_name('predict_y:0')


# freeze the graph
input_graph_path = 'tfdroid.pbtxt'
input_saver_def_path = ""
input_binary = False
checkpoint_path = saved_checkpoint_model
output_node_names = "predict_y"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_tfdroid.pb'
output_optimized_graph_name = 'optimized_frozen_tfdroid.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph=input_graph_path,
                          input_saver=input_saver_def_path,
                          input_binary=input_binary,
                          input_checkpoint=checkpoint_path,
                          output_node_names=output_node_names,
                          restore_op_name=restore_op_name,
                          filename_tensor_name=filename_tensor_name,
                          output_graph=output_frozen_graph_name,
                          clear_devices=clear_devices,
                          initializer_nodes="")


"""
# optimize the model file
input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "r") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_graph_def,
    ["input_x", "seq_len"],  # an array of the input node(s)
    ["predict_y"],  # an array of output nodes
    tf.float32.as_datatype_enum)

# Save the optimized graph
f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())
"""










