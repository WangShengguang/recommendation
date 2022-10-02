"""

https://www.tensorflow.org/guide/keras/save_and_serialize?hl=zh-cn
"""
import collections

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.keras import callbacks


def freeze_to_pb(model, save_path):
    # https://stackoverflow.com/questions/58119155/freezing-graph-to-pb-in-tensorflow2

    def def_input(*args, **kwargs):
        return model(*args, **kwargs)  # model的input参数

    full_model = tf.function(def_input)

    input_spec_dict = collections.OrderedDict()
    for node in model.inputs:
        key = node.name
        input_spec_dict[key] = tf.TensorSpec(node.shape, node.dtype)
    full_model = full_model.get_concrete_function(**input_spec_dict)
    # inputs = [tf.TensorSpec(node.shape, node.dtype)
    #           for node in model.inputs]
    # full_model = full_model.get_concrete_function(inputs)
    full_model = full_model.get_concrete_function(**input_spec_dict)
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./",
                      name=save_path,
                      as_text=False)


class CustomModelCheckpoint(callbacks.ModelCheckpoint):

    def __init__(self, *args, save_pb=False, **kwargs):
        super(CustomModelCheckpoint, self).__init__(*args, **kwargs)
        self._save_pb = save_pb

    def save_to_pb(self, epoch, logs):
        if self._save_pb:
            filepath = self._get_file_path(epoch, logs) + '.pb'
            freeze_to_pb(self.model, save_path=filepath)

    def on_train_batch_end(self, batch, logs=None):
        super(CustomModelCheckpoint, self).on_train_batch_end(batch, logs=logs)
        if self._should_save_on_batch(batch):
            self.save_to_pb(epoch=self._current_epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs=logs)
        if self.save_freq == 'epoch':
            self.save_to_pb(epoch=self._current_epoch, logs=logs)
