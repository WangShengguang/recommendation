"""
https://keras.io/zh/layers/writing-your-own-keras-layers/
https://www.tensorflow.org/guide/estimator?hl=zh-cn
"""
import copy
import typing

from tensorflow.python import keras
from tensorflow.python.keras import layers

from rec.base.feature import get_features, SparseFeature, VarLenSparseFeature


class BaseModel(keras.Model):
    def __init__(self, features, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)
        self.sparse_features, self.dense_features, self.varlen_features = get_features(features)
        self.get_embedding()

    def get_embedding(self, sparse_features=None):
        if sparse_features is None:
            sparse_features = copy.deepcopy(self.sparse_features)
            for varlen_feature in self.varlen_features:
                sparse_features.append(varlen_feature.sparse_feature)
        embedding_dict = {}
        for feature in sparse_features:
            # breakpoint()
            embedding_dict[feature.name] = keras.layers.Embedding(input_dim=feature.vocab_size,
                                                                  output_dim=feature.embedding_dim,
                                                                  name=feature.embedding_name)
        #
        self.embedding_dict = embedding_dict
        # return embedding_dict

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        super(BaseModel, self).build(input_shape)  # 一定要在最后调用它

    def compute_output_shape(self, input_shape):
        """
        如果你的层更改了输入张量的形状，你应该在这里定义形状变化的逻辑，这让Keras能够自动推断各层的形状。
        """
        return input_shape
