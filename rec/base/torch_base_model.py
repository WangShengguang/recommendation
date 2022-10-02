"""
https://keras.io/zh/layers/writing-your-own-keras-layers/
https://www.tensorflow.org/guide/estimator?hl=zh-cn
"""
import copy
import traceback

import torch
from torch import nn

from rec.base.feature import get_features, get_all_sparse_features


class BaseModel(nn.Module):
    def __init__(self, features, *args, **kwargs):
        super(BaseModel, self).__init__()
        self.sparse_features, self.dense_features, self.varlen_features = get_features(features)
        self.get_embedding()

    @property
    def concated_dim(self):
        total_dim = 0
        all_sparse_features = get_all_sparse_features(self.sparse_features + self.varlen_features)
        for spare_feature in all_sparse_features:
            total_dim += spare_feature.embedding_dim
        total_dim += len(self.dense_features)
        return total_dim

    def get_concated_embedding(self, input_map):
        sparse_embeddings = []
        for feature in self.sparse_features:
            embed = self.embedding_dict[feature.name](input_map[feature.name])
            sparse_embeddings.append(embed)
        concat_sparse_embed = torch.cat(sparse_embeddings, dim=-1)  #
        concat_sparse_embed = torch.squeeze(concat_sparse_embed, dim=1)  # （batch, 1, dim) -> (batch, dim)
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
                    embed = torch.mean(embed, dim=-1)
                else:
                    embed = torch.sum(embed, dim=-1)
                var_len_feature_values.append(embed)
            except Exception:
                traceback.print_exc()
                # breakpoint()
        if var_len_feature_values:
            concat_varlen_embed = torch.cat(var_len_feature_values, dim=-1)  #
            merged_embed = torch.cat([concat_sparse_embed, concat_varlen_embed], dim=-1)
        else:
            merged_embed = concat_sparse_embed
        return merged_embed

    def get_embedding(self, sparse_features=None):
        if sparse_features is None:
            sparse_features = copy.deepcopy(self.sparse_features)
            for varlen_feature in self.varlen_features:
                sparse_features.append(varlen_feature.sparse_feature)
        embedding_dict = {}
        for feature in sparse_features:
            # breakpoint()
            embedding_dict[feature.name] = nn.Embedding(num_embeddings=feature.vocab_size,
                                                        embedding_dim=feature.embedding_dim)
        #
        self.embedding_dict = embedding_dict
        # return embedding_dict
