import queue
import time

import torch
from torch.utils.data import Dataset

from datasets.ml1m.movielens_1m import Movielens1M
from datasets.tokenizer import pandas_read
from rec.utils import decorators, file_utils
from rec.utils.sequence import pad_sequence


class MovielensDataset(Dataset):
    def __init__(self, sample_tsv_path=None):
        if sample_tsv_path is None:
            sample_tsv_path = Movielens1M.ml1m_sample_tsv
        self.sample_tsv_path = sample_tsv_path
        #
        self.vocab = Movielens1M.get_vocab()
        self.col2meta = {'item_Genres': {'maxlen': 4, 'col_type': ''},
                         'item_Title': {'maxlen': 50, 'col_type': 'text'},
                         'history': {'maxlen': 100, 'col_type': 'history'}
                         }
        #
        self.init()
        self.prefetch()
        time.sleep(3)

    def init(self):
        self.genres_len = 2
        self.genres_pad_values = [0 for _ in range(self.genres_len)]

    def get_row_sample(self, row):
        sample = {}
        for col, val in row.items():
            if col == 'label':
                v = val
            elif col in ('user_id', 'item_id', 'user_Age',
                         'user_Gender', 'user_Zip-code', 'user_Occupation'):
                v = int(self.vocab[col].get(str(val), 0))
                # v = torch.tensor(v, dtype=torch.int)
            elif col in {'item_Genres', 'item_Title', 'history'}:
                maxlen = self.col2meta[col]['maxlen']
                values = [self.vocab[col].get(v, 0) for v in val.split(',')]
                v = pad_sequence(values, maxlen=maxlen)
                # v = torch.tensor(v, dtype=torch.int)
            else:
                # print(col)
                continue
            sample[col] = v
        return sample

    @decorators.async_task
    def prefetch(self):
        self.queue = queue.Queue(maxsize=10 * 10000)
        while True:
            for data_df in pandas_read(self.sample_tsv_path, sep='\t'):
                data_df.fillna('', inplace=True)
                data_df = data_df.sample(frac=1)
                # data_df = shuffle(data_df)
                for i, row in data_df.iterrows():
                    sample = self.get_row_sample(row)
                    self.queue.put(sample)

    def __len__(self):
        """ 控制样本，dataloader, 根据这个数目来决定，一个epoch的开始结束"""
        if not hasattr(self, 'sample_count'):
            self.sample_count = file_utils.get_num_lines(self.sample_tsv_path)
        return self.sample_count

    def __getitem__(self, idx):
        sample = self.queue.get(timeout=5)
        for k, v in sample.items():
            if k == 'label':
                v = torch.tensor(v, dtype=torch.float)
            else:
                v = torch.tensor(v, dtype=torch.long)
            sample[k] = v
        return sample

    # def collate(self, batches):
    #     all_u, all_v, all_neg_v = [], [], []
    #     for (u, v, neg_v) in batches:
    #         all_u.append(u)
    #         all_v.append(v)
    #         all_neg_v.append(neg_v)
    #     return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)


def main():
    Movielens1M.gen_samples()  # 样本生成
    # Movielens1M.get_vocab()  # 词典生成， 离散值，连续值
    get_tokenizer().build_vocab()


if __name__ == '__main__':
    main()
