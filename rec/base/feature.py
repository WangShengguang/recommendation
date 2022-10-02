class Feature(object):
    def __init__(self, *args, **kwargs):
        # self.name = ''
        # self.vocab_size = 0
        # self.embedding_dim = 0
        pass


class SparseFeature(Feature):
    def __init__(self, name, vocab_size, embedding_dim, ):
        super(SparseFeature, self).__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    @property
    def embedding_name(self):
        return f"embedding_{self.name}"


class DenseFeature(Feature):
    def __init__(self, name, embedding_dim=1):
        super(DenseFeature, self).__init__()
        self.name = name
        self.embedding_dim = embedding_dim


class VarLenSparseFeature(Feature):
    def __init__(self, sparse_feature, combiner, maxlen, col_type=''):
        super(VarLenSparseFeature, self).__init__()
        self.sparse_feature: SparseFeature = sparse_feature
        self.combiner = combiner
        self.maxlen = maxlen
        #
        self.col_type = col_type  # text, history


class TextFeature(Feature):
    def __init__(self, embedding_dim, maxlen):
        super(TextFeature, self).__init__()
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen


def get_features(features):
    sparse_features = []
    dense_features = []
    varlen_features = []
    for feature in features:
        if isinstance(feature, SparseFeature):
            sparse_features.append(feature)
        elif isinstance(feature, DenseFeature):
            dense_features.append(feature)
        elif isinstance(feature, VarLenSparseFeature):
            varlen_features.append(feature)
        else:
            raise TypeError(feature)
    return sparse_features, dense_features, varlen_features


def get_all_sparse_features(features):
    sparse_features = []
    for feature in features:
        if isinstance(feature, SparseFeature):
            sparse_features.append(feature)
        elif isinstance(feature, VarLenSparseFeature):
            sparse_features.append(feature.sparse_feature)
        else:
            raise TypeError(feature)
    return sparse_features
