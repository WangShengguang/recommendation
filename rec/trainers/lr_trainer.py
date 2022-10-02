import os.path
import pathlib
import sys

import tqdm

cur_dir_pather = pathlib.Path(__file__).absolute()
root_dir = cur_dir_pather.parent.parent.parent.as_posix()
sys.path.append(root_dir)
print(root_dir)
# exit()
import tensorflow as tf
# from tensorflow import optimizers
from datasets.tokenizer import Tokenizer
from rec.base.feature import SparseFeature, DenseFeature, VarLenSparseFeature
from datasets.ml1m.movielens_1m import Movielens1M
from datasets.ml1m.ml1m_tfrecords import Movielens_1m_TFRecords
from rec.rank.lr import LR
from rec.utils import file_utils
from tensorflow.python.keras.metrics import BinaryAccuracy, AUC


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
        self.model = LR(features)
        # optimizer = keras.optimizers.Optimizer('')
        # optimizer = keras.optimizers.Optimizer('')
        optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        # optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        # Movielens_1m_TFRecords.samples_to_tfrecords(col2meta=self.col2meta)
        tf_record_dataset = Movielens_1m_TFRecords.get_tf_record_dataset(col2meta=self.col2meta)
        # pbar = tqdm.tqdm(total=file_utils.get_num_lines(), )
        pbar = tqdm.tqdm()
        global_step = 0
        epochs = 10
        # for batch_data, label in tf_record_dataset.batch(batch_size=256).prefetch(tf.data.experimental.AUTOTUNE):
        # for batch_data, label in tf_record_dataset.batch(batch_size=4):
        for batch_data, label in tf_record_dataset.shuffle(buffer_size=10000).repeat(epochs).batch(batch_size=32):
            global_step += 1
            # if global_step >= 3909:
            #     breakpoint()
            # https://www.tensorflow.org/guide/autodiff?hl=zh-cn
            with tf.GradientTape(persistent=True) as tape:
                # 默认情况下GradientTape的资源在调用gradient函数后就被释放，再次调用就无法计算了。所以如果需要多次计算梯度，需要开启persistent=True属性
                # 一般在网络中使用时，不需要显式调用watch函数，使用默认设置，GradientTape会监控可训练变量 (由tf.Variable创建的traiable=True属性（默认）的变量)
                # tape.watch()  # 手动添加监视; 可以添加非可训练的外部变量(tensor)
                batch_data['label'] = label
                # breakpoint()
                # y_pred, loss = self.model(batch_data, training=True)
                y_pred, loss = self.model.call(batch_data, training=True)
                # with tape.stop_recording():
                #     # The gradient compilerutation below is not traced, saving memory.
                # breakpoint()
                #
                # breakpoint()
                acc = BinaryAccuracy()(label, y_pred)
                auc = AUC()(label, y_pred)
                pbar.set_description(f"epochs: {epochs}, global_step: {global_step},"
                                     f" acc:{acc.numpy():.04f},  auc:{auc.numpy():.04f},  loss: {loss.numpy():.04f}")
                pbar.update()
                if global_step % 100 == 0:
                    print()
                # print(batch_data)
                # breakpoint()
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))
            # breakpoint()
            # if global_step >= 3907:
            #     breakpoint()

    def train_keras(self):
        features = self.get_features()
        model = LR(features)
        # adam_optimizer = tf.python.keras.Adam(learning_rate=1e-3, epsilon=1e-07, clipvalue=5.0)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[
                          tf.keras.metrics.BinaryCrossentropy(),
                          tf.keras.metrics.AUC()
                      ]
                      )
        tf_record_dataset = Movielens_1m_TFRecords.get_tf_record_dataset(col2meta=self.col2meta)
        train_data = tf_record_dataset.batch(batch_size=128).prefetch(tf.data.experimental.AUTOTUNE)
        model.fit(train_data,
                  epochs=100,
                  # callbacks=[cp_callback],
                  # validation_split=0.1,
                  # batch_size=128,
                  verbose=2)


def main():
    LRTrainer().train()
    # LRTrainer().train_keras()


if __name__ == '__main__':
    main()
