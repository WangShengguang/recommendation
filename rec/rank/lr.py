"""
https://stackoverflow.com/questions/51672903/keras-tensorflow-how-to-set-breakpoint-debug-in-custom-layer-when-evaluating

"""
import traceback
from typing import List

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from rec.base.base_model import BaseModel
from rec.base.feature import Feature


# pad_sequences

class LR(BaseModel):
    def __init__(self, features: List[Feature]):
        super(LR, self).__init__(features)
        #
        # self.linear = keras.layers.Dense(units=1)
        self.mlp = keras.models.Sequential([
            # keras.layers.Dense(128),
            # keras.layers.ReLU(),
            keras.layers.Dense(64),
            keras.layers.ReLU(),
            keras.layers.Dense(units=1)]
        )
        #
        # self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
        # self.cross_entropy = keras.losses.CategoricalCrossentropy()
        # self.cross_entropy = keras.losses.SparseCategoricalCrossentropy()

    # def build(self, input_shape):
    #     super(LR, self).build(input_shape)

    def call(self, input_map, training=None, mask=None):
        # sparse feature
        sparse_embeddings = []
        for feature in self.sparse_features:
            embed = self.embedding_dict[feature.name](input_map[feature.name])
            sparse_embeddings.append(embed)
        concat_sparse_embed = tf.concat(sparse_embeddings, axis=-1)  #
        concat_sparse_embed = tf.squeeze(concat_sparse_embed, axis=1)  # （batch, 1, dim) -> (batch, dim)
        # dense feature
        dense_values = []
        for feature in self.dense_features:
            dense_values.append(input_map[feature.name])
        # var_len feature
        var_len_feature_values = []
        for var_len_feature in self.varlen_features:
            try:
                _feature = var_len_feature.sparse_feature
                embed = self.embedding_dict[_feature.name](input_map[_feature.name])
                if var_len_feature.combiner == 'average':
                    embed = tf.reduce_mean(embed, axis=-1)
                else:
                    embed = tf.reduce_sum(embed, axis=-1)
                var_len_feature_values.append(embed)
            except Exception:
                traceback.print_exc()
                # breakpoint()
        if var_len_feature_values:
            concat_varlen_embed = tf.concat(var_len_feature_values, axis=-1)  #
            merged_embed = tf.concat([concat_sparse_embed, concat_varlen_embed], axis=-1)
        else:
            merged_embed = concat_sparse_embed
        #
        logits = self.mlp(merged_embed)
        if 'label' in input_map:
            label = input_map['label']
            # breakpoint()
            y_pred = tf.sigmoid(logits)
            loss = self.cross_entropy(y_pred, label)
            # breakpoint()
            return y_pred, loss
        y_pred = tf.sigmoid(logits)
        return y_pred
