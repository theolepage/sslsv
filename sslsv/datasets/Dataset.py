from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset

from sslsv.datasets.Sampler import SamplerConfig
from sslsv.datasets.DataAugmentation import DataAugmentation, DataAugmentationConfig
from sslsv.datasets.utils import load_audio


class FrameSamplingEnum(Enum):

    DEFAULT = "default"
    DINO = "dino"


@dataclass
class DatasetConfig:

    ssl: bool = False
    augmentation: DataAugmentationConfig = None
    sampler: SamplerConfig = None
    frame_length: int = 32000
    frame_sampling: FrameSamplingEnum = FrameSamplingEnum.DEFAULT
    max_samples: int = None
    train: str = "voxceleb2_train.csv"
    label_key: str = "Speaker"
    base_path: Path = Path("./data/")
    num_workers: int = 8
    pin_memory: bool = False


class Dataset(TorchDataset):

    def __init__(
        self,
        config,
        files,
        labels=None,
        num_frames=1,
    ):
        super().__init__()

        self.config = config
        self.files = files
        self.labels = labels
        self.num_frames = num_frames

        self.augmentation = None
        if self.config.augmentation and self.config.augmentation.enable:
            self.augmentation = DataAugmentation(
                self.config.augmentation, self.config.base_path
            )

    def __len__(self):
        if self.config.max_samples:
            return min(len(self.files), self.config.max_samples)
        return len(self.files)

    def preprocess_data(self, data, augment=True):
        if augment and self.augmentation:
            assert data.ndim == 2 and data.shape[0] == 1  # (1, T)
            data = self.augmentation(data)
        return data

    def __getitem__(self, i):
        data = load_audio(
            self.config.base_path / self.files[i],
            frame_length=self.config.frame_length,
            num_frames=self.num_frames,
        )  # (N, T)

        x = torch.FloatTensor(self.preprocess_data(data)).squeeze(0)

        info = {"files": self.files[i]}
        if self.labels:
            info.update({"labels": self.labels[i]})

        return i, x, info
