from collections import defaultdict
from typing import Iterable

import numpy as np

from torch.utils.data import Sampler as TorchSampler

from sslsv.datasets.Dataset import Dataset, DatasetConfig
from sslsv.datasets.Sampler import Sampler, SamplerConfig
from sslsv.datasets.DistributedSamplerWrapper import DistributedSamplerWrapper

from .test_dataset import get_files_labels


def test_nb_samples_per_spk():
    files, labels = get_files_labels("data/voxceleb1_train.csv")

    dataset = Dataset(DatasetConfig(), files, labels)

    config = SamplerConfig(nb_samples_per_spk=5)
    sampler = Sampler(dataset, 64, config)

    d = defaultdict(int)
    for i in iter(sampler):
        d[labels[i]] += 1
    for count in d.values():
        assert count <= 5


def test_create_contrastive_pairs():
    files, labels = get_files_labels("data/voxceleb1_train.csv")

    dataset = Dataset(DatasetConfig(), files, labels)

    config = SamplerConfig(create_contrastive_pairs=True)
    sampler = Sampler(dataset, 64, config)

    for i in iter(sampler):
        assert isinstance(i, tuple)
        assert labels[i[0]] == labels[i[1]]


def test_prevent_class_collisions(batch_size=64):
    files, labels = get_files_labels("data/voxceleb1_train.csv")

    dataset = Dataset(DatasetConfig(), files, labels)

    config = SamplerConfig(prevent_class_collisions=True)
    sampler = Sampler(dataset, batch_size, config)

    idx = list(iter(sampler))
    labels = np.array(labels)[idx]

    for i in range(len(idx)):
        batch_start_i = i - i % batch_size
        batch_end_i = batch_start_i + batch_size
        nb_unique_labels = len(np.unique(labels[batch_start_i:batch_end_i]))
        assert nb_unique_labels == batch_size


class DummySampler(TorchSampler):

    def __init__(self, count: int):
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterable[int]:
        return iter(np.arange(self.count))


def test_distributed_wrapper_basic():
    sampler = DummySampler(count=5)

    sampler_0 = DistributedSamplerWrapper(sampler, world_size=2, rank=0)
    sampler_1 = DistributedSamplerWrapper(sampler, world_size=2, rank=1)

    assert list(iter(sampler_0)) == [0, 2, 4]
    assert list(iter(sampler_1)) == [1, 3, 0]
