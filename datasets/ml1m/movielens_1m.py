import os
import pathlib

import pandas as pd

from datasets.tokenizer import Tokenizer

cur_dir_pather = pathlib.Path(__file__).absolute().parent
root_dir = cur_dir_pather.parent.parent.as_posix()
data_dir = os.path.join(root_dir, 'datasets/data')
print("data_dir: {}".format(data_dir))


class Movielens1M(object):
    ml1m_sample_tsv = cur_dir_pather.joinpath('ml1m_sample.tsv').as_posix()
    # samples
    spare_columns = ['user_id', 'item_id', 'user_Gender', 'user_Age', 'user_Occupation', 'user_Zip-code']
    dense_columns = []
    varlen_columns = ['item_Genres', 'item_Title']
    # text_columns = ['item_Title']
    y_columns = ['label']

    @classmethod
    def get_vocab(cls):
        tokenizer = Tokenizer(sample_tsv_path=cls.ml1m_sample_tsv,
                              sparse_column_names=cls.spare_columns,
                              dense_column_names=cls.dense_columns,
                              varlen_column_names=cls.varlen_columns,
                              # text_column_names=cls.text_columns
                              )
        tokenizer.build_vocab()
        return tokenizer.get_vocab()

    @classmethod
    def read_ml_1m_data(cls):
        ml_1m_dir = os.path.join(data_dir, 'ml-1m')
        #
        rating_df = pd.read_csv(os.path.join(ml_1m_dir, 'ratings.dat'),
                                sep='::',
                                # names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                                names=['user_id', 'item_id', 'rating', 'timestamp'],
                                dtype={'Rating': int})
        # ZIP-Code是美國郵政使用的一種邮政编码

        user_df = pd.read_csv(os.path.join(ml_1m_dir, 'users.dat'),
                              sep='::',
                              # names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                              names=['user_id', 'user_Gender', 'user_Age', 'user_Occupation', 'user_Zip-code'],
                              )
        # Genres 风格，类型
        item_df = pd.read_csv(os.path.join(ml_1m_dir, 'movies.dat'),
                              sep='::',
                              # names=['MovieID', 'Title', 'Genres'],
                              names=['item_id', 'item_Title', 'item_Genres'],
                              encoding="ISO-8859-1"
                              )
        return user_df, item_df, rating_df

    @classmethod
    def gen_sample_df(cls):
        user_df, item_df, rating_df = cls.read_ml_1m_data()

        #
        def get_label(rating):
            label = 1 if int(rating) >= 4 else 0
            return label

        sample_df = rating_df.join(item_df.set_index('item_id'), on='item_id', how='left')
        sample_df = sample_df.join(user_df.set_index('user_id'), on='user_id', how='left')
        sample_df['label'] = sample_df['rating'].apply(get_label)
        sample_df.to_csv(cls.ml1m_sample_tsv, sep='\t', index=False, encoding='utf_8_sig')
        return sample_df


def main():
    Movielens1M.gen_sample_df()  # 样本生成


if __name__ == '__main__':
    main()
