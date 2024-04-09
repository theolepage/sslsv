from pathlib import Path
import pandas as pd
import torch

from sslsv.Config import Config
from sslsv.datasets.Dataset import Dataset, DatasetConfig, FrameSamplingEnum
from sslsv.datasets.SSLDataset import SSLDataset
from sslsv.utils.helpers import load_train_dataloader


def get_files_labels(path):
    df = pd.read_csv(path)
    files = df["File"].tolist()
    labels = pd.factorize(df["Speaker"])[0].tolist()
    return files, labels


def test_ssl_vox1():
    files, labels = get_files_labels("data/voxceleb1_train.csv")

    assert len(files) == 148642
    assert len(labels) == 148642

    # 64637: voxceleb1/id10572/TenWR96X_0o/00001.wav,id10572
    assert files[64637 - 2] == "voxceleb1/id10572/TenWR96X_0o/00001.wav"
    assert labels[64637 - 2] == 531

    dataset = SSLDataset(DatasetConfig(), files, labels)

    assert len(dataset) == 148642

    idx, X, info = dataset[0]
    assert isinstance(X, torch.Tensor)
    assert X.dtype == torch.float32
    assert X.size() == (2, 32000)


def test_vox1():
    files, labels = get_files_labels("data/voxceleb1_train.csv")

    dataset = Dataset(DatasetConfig(), files, labels)

    assert len(dataset) == 148642

    idx, X, info = dataset[0]
    assert isinstance(X, torch.Tensor)
    assert X.dtype == torch.float32
    assert X.size() == (32000,)


def test_max_samples():
    files, labels = get_files_labels("data/voxceleb1_train.csv")

    dataset = Dataset(DatasetConfig(max_samples=1359), files, labels)

    assert len(dataset) == 1359


def test_ssl_vox2():
    files, labels = get_files_labels("data/voxceleb2_train.csv")
    assert len(files) == 1092009
    assert len(labels) == 1092009

    # 580949: voxceleb2/id04981/9qJOtRs4tOI/00002.wav,id04981
    assert files[580949 - 2] == "voxceleb2/id04981/9qJOtRs4tOI/00002.wav"
    assert labels[580949 - 2] == 3240

    dataset = SSLDataset(DatasetConfig(), files, labels)

    assert len(dataset) == 1092009

    idx, X, info = dataset[0]
    assert isinstance(X, torch.Tensor)
    assert X.dtype == torch.float32
    assert X.size() == (2, 32000)


def test_load_train_dataloader():
    config = Config()
    train_dataloader = load_train_dataloader(config)

    assert len(train_dataloader) == 1092009 // config.trainer.batch_size
