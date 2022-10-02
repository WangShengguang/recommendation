import torch.utils.data.datapipes as dp
from torchrec.datasets.movielens import movielens_20m

datapipe = movielens_20m("/home/disk/recommendation/data/ml-25m")
datapipe = dp.Batch(datapipe, 100)
datapipe = dp.Collate(datapipe)
batch = next(iter(datapipe))
