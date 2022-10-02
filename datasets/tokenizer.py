import collections
import heapq
import os.path
import pathlib
import sys
import traceback
import typing

import numpy as np
import pandas as pd
import tqdm

cur_dir_pather = pathlib.Path(__file__).absolute().parent
root_dir = cur_dir_pather.parent.parent.parent.as_posix()

sys.path.append(root_dir)
print(f"dataset root_dir: {root_dir}")
# exit()
from rec.utils import file_utils


def pandas_read(file_path, sep=',',
                names=None, usecols=None,
                col2dtype=None, chunksize=10 * 10000, **kwargs) -> typing.List[pd.DataFrame]:
    iter_df = pd.read_csv(file_path,
                          names=names,
                          usecols=usecols,
                          sep=sep,
                          chunksize=chunksize,
                          dtype=col2dtype,  # 12 min
                          # engine='python',  # c or python
                          # error_bad_lines=False,  # 需要配合 engine='python', Fix Lines with too many fields
                          # converters=converters,  # 34min
                          # quotechar='"',  # 不能解决_csv.Error: '' expected after '"'
                          # index_col='id' # ValueError: Index id invalid
                          **kwargs
                          )
    pbar = tqdm.tqdm(iter_df,
                     total=file_utils.get_num_lines(file_path) // chunksize + 1,
                     desc=f"read_csv {pathlib.Path(file_path).name}")
    for data_df in pbar:
        yield data_df


class Tokenizer(object):

    def __init__(self, sample_tsv_path,
                 sparse_column_names,
                 dense_column_names,
                 varlen_column_names,
                 # text_column_names
                 ):
        """
        sparse_columns = ['col1', 'col2']
        dense_columns = ['col1', 'col2']
        varlen_columns = [ ('col1', ','), ('col2', '\t')]
        """
        self.sample_tsv_path = sample_tsv_path
        #
        self.meta_path = self.sample_tsv_path + '.meta'
        self.values_path = self.sample_tsv_path + '.values'
        self.vocab_path = self.sample_tsv_path + '.vocab'
        #
        self.sparse_column_names = sparse_column_names
        self.dense_column_names = dense_column_names
        self.varlen_column_names = varlen_column_names
        #
        self.varlen_column_to_sep_map = {}
        for col in varlen_column_names:
            self.varlen_column_to_sep_map[col] = ','
        # for col in text_column_names:
        #     self.varlen_column_to_sep_map[col] = ' '
        #

    def get_data_info(self):
        col2unique_vals = collections.defaultdict(lambda: collections.Counter())
        col2meta = {}
        for col in self.sparse_column_names:
            col2meta[col] = {}
        for col in self.dense_column_names:
            col2meta[col] = {'max': -np.inf, 'min': np.inf}
        for col in self.varlen_column_to_sep_map:
            col2meta[col] = {'max_len': -1, 'min_len': 100 * 10000}
            # col2meta[f"{col}_len"] = {}

        #
        def read_data(sep='\t'):
            for data_df in pandas_read(self.sample_tsv_path, sep=sep):
                data_df.fillna('', inplace=True)
                for i, row in data_df.iterrows():
                    yield row

        #
        for row in read_data():
            for col in self.sparse_column_names:
                col2unique_vals[col].update([row[col]])
            #
            for col, sep in self.varlen_column_to_sep_map.items():
                try:
                    if sep is not None:
                        vals = row[col].split(sep)
                    else:
                        vals = [token for token in row[col]]
                except Exception:
                    vals = []
                col2unique_vals[col].update(vals)
                #
                val_len = len(vals)
                # col2meta[col]['max_len'] = max(val_len, col2meta[col]['max_len'])
                # col2meta[col]['min_len'] = min(val_len, col2meta[col]['min_len'])
                col2unique_vals[f'{col}_len'].update([val_len])
                # if col=='item_Title':
                #     breakpoint()
            #
            for col in self.dense_column_names:
                v = float(row[col])
                # col2unique_vals[col]['max'] = max(v, col2unique_vals[col]['max'])
                # col2unique_vals[col]['min'] = min(v, col2unique_vals[col]['min'])
                col2unique_vals[col].update([v])
            # breakpoint()
        #
        for col in self.sparse_column_names:
            col2meta[col]['vocab_size'] = len(col2unique_vals[col])
        for col in self.dense_column_names:
            col2meta[col]['max'] = max(col2unique_vals[col].keys())
            col2meta[col]['min'] = min(col2unique_vals[col].keys())
            kv_pairs = heapq.nlargest(n=5, iterable=col2unique_vals[col].items(), key=lambda kv: kv[1])
            total = sum(col2unique_vals[col].values())
            col2meta[col]['most_freq'] = [[k, v, f"{v / total}:.3%"] for k, v in kv_pairs]
        for col in self.varlen_column_to_sep_map:
            col2meta[col]['max_len'] = max(col2unique_vals[f"{col}_len"].keys())
            col2meta[col]['min_len'] = min(col2unique_vals[f"{col}_len"].keys())
            kv_pairs = heapq.nlargest(n=5, iterable=col2unique_vals[f"{col}_len"].items(), key=lambda kv: kv[1])
            total = sum(col2unique_vals[col].values())
            col2meta[col]['most_freq_len'] = [[k, v, f"{v / total}:.3%"] for k, v in kv_pairs]
        #
        file_utils.json_dump(dict(col2meta), self.meta_path)
        file_utils.json_dump(dict(col2unique_vals), self.values_path)

    def build_vocab(self):
        self.get_data_info()
        # col2meta = file_utils.json_load(self.file_path + '.meta')
        col2vals = file_utils.json_load(self.values_path)
        vocab = {}
        for col in self.sparse_column_names + [col for col in self.varlen_column_to_sep_map]:
            vocab[col] = {'UNK': 0}
            for val, count in sorted(col2vals[col].items(), key=lambda kv: kv[1]):
                vocab[col][val] = len(vocab[col])
        file_utils.json_dump(vocab, self.vocab_path)

    def get_vocab(self):
        if not os.path.isfile(self.vocab_path):
            self.build_vocab()
        vocab = file_utils.json_load(self.vocab_path)
        return vocab

    def run(self):
        # self.get_data_info()
        self.build_vocab()


def main():
    # 词典生成， 离散值，连续值
    # Movielens1M.gen_samples()
    from datasets.ml1m.movielens_1m import Movielens1M
    dataset = Tokenizer(sample_tsv_path=Movielens1M.ml1m_sample_tsv,
                        sparse_column_names=Movielens1M.spare_columns,
                        dense_column_names=Movielens1M.dense_columns,
                        varlen_column_names=Movielens1M.varlen_columns,
                        # text_column_names=Movielens1M.text_columns
                        )
    # dataset.get_data_info()
    dataset.build_vocab()


if __name__ == '__main__':
    main()
