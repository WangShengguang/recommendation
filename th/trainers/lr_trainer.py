import os.path
import pathlib
import sys

import numpy as np
import torch
import tqdm

cur_dir_pather = pathlib.Path(__file__).absolute()
root_dir = cur_dir_pather.parent.parent.parent.as_posix()
sys.path.append(root_dir)
print(root_dir)
# exit()
# from tensorflow import optimizers
from datasets.tokenizer import Tokenizer
from rec.base.feature import SparseFeature, DenseFeature, VarLenSparseFeature
from datasets.ml1m.movielens_1m import Movielens1M
from th.rank.mlp import MLP
from rec.utils import file_utils
from torch.utils.data import DataLoader
from th.datasets.ml1m_torch_dataset import MovielensDataset


class LRTrainer(object):
    def __init__(self):
        self.col2meta = {'item_Genres': {'maxlen': 4, 'col_type': ''},
                         'item_Title': {'maxlen': 50, 'col_type': 'text'},
                         'history': {'maxlen': 100, 'col_type': 'history'}
                         }
        self.data_prepare(overwrite=False)

    def data_prepare(self, overwrite=False):
        if overwrite or not os.path.isfile(Movielens1M.ml1m_sample_tsv):
            Movielens1M.gen_samples()
        self.tokenizer = Tokenizer(sample_tsv_path=Movielens1M.ml1m_sample_tsv,
                                   sparse_column_names=Movielens1M.spare_columns,
                                   dense_column_names=Movielens1M.dense_columns,
                                   varlen_column_names=Movielens1M.varlen_columns,
                                   # text_column_names=Movielens1M.text_columns
                                   )
        if overwrite or not os.path.isfile(self.tokenizer.values_path):
            self.tokenizer.get_data_info()
        if overwrite or not os.path.isfile(self.tokenizer.vocab_path):
            self.tokenizer.build_vocab()
        # Movielens_1m_TFRecords.gen_tfrecords()

    def get_features(self, spare_columns=None, varlen_columns=None, dense_columns=None):
        if spare_columns is None:
            spare_columns = Movielens1M.spare_columns
            spare_columns = ['user_id', 'item_id', 'user_Gender', 'user_Age', 'user_Occupation', 'user_Zip-code']
            spare_columns = ['user_id', 'item_id', 'user_Gender', 'user_Age']
            # spare_columns = ['user_id', 'item_id']
        if varlen_columns is None:
            varlen_columns = Movielens1M.varlen_columns
            varlen_columns = ['history']  # ['item_Genres', 'history']
            varlen_columns = []
        if dense_columns is None:
            dense_columns = []
        #
        vocab = file_utils.json_load(self.tokenizer.vocab_path)
        name2features = {}
        for col in spare_columns + varlen_columns:
            vocab_size = len(vocab[col])
            embedding_dim = int(vocab_size ** 0.25)
            if embedding_dim > 64:
                embedding_dim = 64
            elif embedding_dim < 2:
                embedding_dim = 2
            # dim = max(min(embedding_dim, 64), 2)  # 2 <= dim <= 64
            _feature = SparseFeature(name=col, vocab_size=vocab_size, embedding_dim=embedding_dim)
            name2features[col] = _feature
        #
        for col in varlen_columns:
            _feature = name2features.pop(col)
            name2features[col] = VarLenSparseFeature(sparse_feature=_feature,
                                                     combiner='average',
                                                     maxlen=self.col2meta[col]['maxlen'],
                                                     col_type=self.col2meta[col]['col_type']
                                                     )
        #
        for col in dense_columns:
            _feature = DenseFeature(name=col)
            name2features[col] = _feature
        #
        features = [_feature for name, _feature in name2features.items()]
        return features

    def train(self):
        features = self.get_features()
        model = MLP(features)
        dataset = MovielensDataset(sample_tsv_path=Movielens1M.ml1m_sample_tsv)

        data_loader = DataLoader(dataset=dataset, batch_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        pbar = tqdm.tqdm(total=len(dataset) // 32)
        global_step = 0
        for epoch in range(1, 10 + 1):
            for batch in data_loader:
                # breakpoint()
                global_step += 1
                pred, loss = model(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # breakpoint()
                if global_step % 100 == 0:
                    print()
                pred_label = (pred.squeeze(-1) > 0.5).numpy()
                acc = np.mean(pred_label == batch['label'].numpy())
                pbar.set_description(f"epoch: {epoch}, global_step: {global_step}, "
                                     f"loss: {loss:.04f}, acc: {acc:.04f}")
                pbar.update()
                # breakpoint()


def main():
    LRTrainer().train()


if __name__ == '__main__':
    main()
