import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_sample_df():
    # user_df, item_df, rating_df = Movielens1M.read_ml_1m_data()
    # sample_df = Movielens1M.gen_sample_df()
    # user_id	item_id	rating	timestamp	item_Title	item_Genres	user_Gender	user_Age	user_Occupation	user_Zip-code	label
    #
    ml_1m_dir = os.path.join(root_dir, 'datasets/data/ml-1m')
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

    #
    rating_df['label'] = rating_df['rating'].apply(lambda x: 1 if int(x) >= 4 else 0)
    # sample_df = rating_df[['user_id', 'item_id', 'label']]
    #
    ml1m_sample_tsv = './ml1m_sample.tsv'
    sample_df = rating_df.join(item_df.set_index('item_id'), on='item_id', how='left')
    sample_df = sample_df.join(user_df.set_index('user_id'), on='user_id', how='left')
    sample_df.to_csv(ml1m_sample_tsv, sep='\t', index=False, encoding='utf_8_sig')
    #
    return sample_df


def get_vocab(data_df: pd.DataFrame):
    vocab = defaultdict(Counter)
    for i, row in data_df.iterrows():
        for k, v in row.items():
            vocab[k].update([v])
    #
    real_vocab = {}
    for col, _vocab_count in vocab.items():
        values = sorted(_vocab_count.items(), key=lambda kv: kv[1], reverse=True)
        values = ['UNK'] + [k for k, v in values]
        real_vocab[col] = {v: i for i, v in enumerate(values)}
    # breakpoint()
    return real_vocab


class MovielensDataset(Dataset):
    def __init__(self, sample_df, vocab):
        self.samples = self.get_samples(sample_df, vocab)

    def get_samples(self, sample_df, vocab):
        samples = []
        sparse_cols = ['user_id', 'item_id', 'user_Gender', 'user_Age', 'user_Occupation',
                       'user_Zip-code']
        var_len_cols = ['item_Title', ]
        # feature_cols = ['user_id', 'item_id']
        label_cols = ['label']
        for i, row in sample_df.iterrows():
            sample = []
            for col in sparse_cols:
                v = vocab[col].get(row[col], 0)
                sample.append(v)

            for col in label_cols:
                sample.append(row[col])
            samples.append(sample)
            # try:
            #
            # except Exception:
            #     print(traceback.format_exc())
            #     breakpoint()

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        (user_id, item_id, user_gender, user_Age, user_Occupation,
         user_Zip_code, label) = sample
        user_id = torch.tensor(user_id, dtype=torch.long)
        item_id = torch.tensor(item_id, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)
        return user_id, item_id, label


class LR(nn.Module):
    def __init__(self, user_size_dim_pair, item_size_dim_pair):
        super().__init__()
        user_vocab_size, user_dim = user_size_dim_pair
        self.user_emb = nn.Embedding(user_vocab_size, user_dim)
        item_vocab_size, item_dim = item_size_dim_pair
        self.item_emb = nn.Embedding(item_vocab_size, item_dim)
        #
        self.mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

    def forward(self, batch_user_id, batch_item_id, label=None):
        # batch_user_id, batch_item_id = sample
        user_emb = self.user_emb(batch_user_id)
        # breakpoint()
        item_emb = self.item_emb(batch_item_id)
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        pred = self.mlp(concat_emb)
        if label is not None:
            # breakpoint()
            loss = self.loss_fn(pred.squeeze(-1), label)
            return pred, loss
        return pred


class Trainer(object):
    def __init__(self):
        pass

    def get_data(self):
        pass

    def evaluate(self, model, valid_data_loader):
        all_valid_labels = []
        all_valid_pred_labels = []
        for batch in valid_data_loader:
            *sample, label = batch
            pred = model(*sample)
            pred_labels = (pred.squeeze(-1) > 0.5).numpy().tolist()
            all_valid_pred_labels.extend(pred_labels)
            all_valid_labels.extend(label.numpy().tolist())
        #
        all_valid_pred_labels = np.asarray(all_valid_pred_labels)
        all_valid_labels = np.asarray(all_valid_labels)
        auc_score = roc_auc_score(all_valid_labels, all_valid_pred_labels, average='macro')
        acc = np.mean(all_valid_labels == all_valid_pred_labels)
        return auc_score, acc

    def train(self):
        sample_df = get_sample_df()
        vocab = get_vocab(sample_df)
        #
        train_df, valid_df = train_test_split(sample_df, test_size=0.1)
        train_dataset = MovielensDataset(train_df, vocab)
        valid_dataset = MovielensDataset(valid_df, vocab)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=32)
        valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=32)
        #
        model = LR(user_size_dim_pair=[len(vocab['user_id']), int(len(vocab['user_id']) ** 0.25)],
                   item_size_dim_pair=[len(vocab['item_id']), int(len(vocab['item_id']) ** 0.25)],
                   )
        #
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        #
        batch_size = 32
        per_epoch_step = len(train_dataset) // batch_size + 1
        global_step = 0
        summaryWriter = SummaryWriter("./runs/")
        for epoch in range(1, 10 + 1):
            pbar = tqdm.tqdm(train_data_loader, total=per_epoch_step)
            for cur_epoch_step, batch in enumerate(pbar):
                # breakpoint()
                global_step += 1
                pred, loss = model(*batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # breakpoint()
                pred_label = (pred.squeeze(-1) > 0.5).numpy()
                acc = np.mean(pred_label == batch[-1].numpy())
                # auc_score = auc(pred_label, batch[-1].numpy())
                # auc_score = roc_auc_score(pred_label, batch[-1].numpy(), average='macro', labels=[1, 0])
                pbar.set_description(f"epoch: {epoch}, cur_epoch_step: {cur_epoch_step}/{per_epoch_step}, "
                                     f"loss: {loss:.04f}, acc: {acc:.04f} ")
                pbar.update()
                summaryWriter.add_scalar(tag="training/loss", scalar_value=loss, global_step=global_step)
                summaryWriter.add_scalar(tag="training/acc", scalar_value=acc, global_step=global_step)
                if global_step % 1000 == 0:
                    print()
            #
            valid_auc, valid_acc = self.evaluate(model, valid_data_loader=valid_data_loader)
            print(f"evaluate, epoch: {epoch}, global_step: {global_step}, "
                  f"acc: {valid_acc:.04f}, auc: {valid_auc:.04f}")


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
