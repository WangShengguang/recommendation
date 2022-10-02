"""

给pytorch 读取数据加速 - 体hi的文章 - 知乎 https://zhuanlan.zhihu.com/p/72956595

"""

import pathlib
import sys

import tensorflow as tf

from datasets.tokenizer import pandas_read
from rec.utils.sequence import pad_sequence

cur_dir_pather = pathlib.Path(__file__).absolute().parent
root_dir = cur_dir_pather.parent.parent.parent.as_posix()
sys.path.append(root_dir)
# print(root_dir)
# exit()

from datasets.ml1m.movielens_1m import Movielens1M, get_tokenizer
from rec.utils import file_utils


def _int64_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def _float64_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))


class Movielens_1m_TFRecords(object):
    train_tf_record_path = Movielens1M.ml1m_sample_tsv + '.tfrecord'

    tokenizer = get_tokenizer()
    col2vacab = None

    @classmethod
    def line2sample(cls, row, col2meta=None):
        if cls.col2vacab is None:
            vocab = file_utils.json_load(cls.tokenizer.vocab_path)
            cls.col2vacab = vocab
        # (label, user_id, item_id,
        #  user_Gender, user_Age, user_Occupation, user_Zip_code,
        #  item_Title, item_Genres) = line.rstrip('\n').split('\t')
        # fields = line.rstrip('\n').split('\t')
        # """ 2. 定义features """
        feature = {}
        for col, val in row.items():
            if col == 'label':
                feature[col] = _float64_feature(int(val))
                continue
            col_vocab = cls.col2vacab[col]
            if col in ('user_id', 'item_id', 'user_Age', 'user_Occupation'):
                v = int(col_vocab.get(str(val), 0))
                feature[col] = _int64_feature(v)
            elif col == 'item_Genres':
                values = [col_vocab.get(v, 0) for v in val.split(',')]
                values = pad_sequence(values, maxlen=4)
                feature[col] = _int64_feature(values)
            elif col == 'item_Title':
                values = [col_vocab.get(v, 0) for v in val.split(' ')]
                values = pad_sequence(values, maxlen=50)
                feature[col] = _int64_feature(values)
            elif col == 'history':
                values = [col_vocab.get(v, 0) for v in val.split(',')]
                values = pad_sequence(values, maxlen=100)
                feature[col] = _int64_feature(values)
            else:
                v = int(cls.col2vacab[col][str(val)])
                feature[col] = _int64_feature(v)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # """ 3. 序列化,写入"""
        serialized = example.SerializeToString()
        return serialized

    @classmethod
    def gen_tfrecords(cls, col2meta=None):
        writer = tf.io.TFRecordWriter(cls.train_tf_record_path)  # 1. 定义 writer对象
        # for data_df in pandas_read(Movielens1M.ml1m_sample_tsv, sep=sep):
        #     data_df.fillna('', inplace=True)
        #     for i, row in data_df.iterrows():
        def read_data(sep='\t'):
            for data_df in pandas_read(cls.tokenizer.sample_tsv_path, sep=sep):
                data_df.fillna('', inplace=True)
                for i, row in data_df.iterrows():
                    yield row

        for row in read_data():
            serialized = cls.line2sample(row, col2meta)
            writer.write(serialized)
        writer.close()

    @classmethod
    def get_tf_record_dataset(cls, tf_record_path=None, col2meta=None):
        """
        https://www.ritchie.top/2019/12/25/tensorflow-tfrecord-process/
        https://stackoverflow.com/questions/49303136/tensorflow-tfrecords-for-batch-with-variable-length-data-in-each-example
        :param tf_record_path:
        :return:
        """
        if tf_record_path is None:
            tf_record_path = cls.train_tf_record_path
        features = {}
        for col in Movielens1M.all_sample_columns():
            # if col in {'item_Genres', 'item_Title', 'history'}:
            if col in {'item_Genres', 'item_Title', 'history'}:
                maxlen = col2meta[col]['maxlen']
                # features[col] = tf.io.VarLenFeature(dtype=tf.int64)
                # maxlen = 10
                features[col] = tf.io.FixedLenFeature(shape=[maxlen, ], dtype=tf.int64,
                                                      default_value=[0 for _ in range(maxlen)])
            elif col == 'label':
                features[col] = tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32,
                                                      default_value=0)

            else:
                features[col] = tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64,
                                                      default_value=0)
            # features[col] = tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=None)

        def parse_sample(serialized_example):
            feature_dict = tf.io.parse_single_example(
                serialized_example,
                features=features)
            y = feature_dict.pop('label')
            return feature_dict, y

        parse_dataset = tf.data.TFRecordDataset(filenames=[tf_record_path]).map(parse_sample)
        return parse_dataset

    # def iter(self, batch_size=4):
    #     for data in self


def main():
    # Movielens1M.gen_samples()  # 样本生成
    # 词典生成， 离散值，连续值
    # dataset = get_dataset()
    # dataset.get_data_info()
    # dataset.build_vocab()
    # Movielens_1m_TFRecords.samples_to_tfrecords()
    # https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset?hl=zh-cn
    for data in Movielens_1m_TFRecords.get_tf_record_dataset():
        print(data)
        breakpoint()


if __name__ == '__main__':
    main()
