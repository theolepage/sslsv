from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset

from sslsv.datasets.Sampler import SamplerConfig
from sslsv.datasets.DataAugmentation import DataAugmentation, DataAugmentationConfig
from sslsv.datasets.utils import load_audio


class FrameSamplingEnum(Enum):

    DEFAULT = 'default'
    DINO    = 'dino'


@dataclass
class DatasetConfig:

    ssl            : bool = False
    augmentation   : DataAugmentationConfig = None
    sampler        : SamplerConfig = None
    frame_length   : int = 32000
    frame_sampling : FrameSamplingEnum = FrameSamplingEnum.DEFAULT
    max_samples    : int = None
    train          : str = 'voxceleb2_train.csv'
    label_key      : str = 'Speaker'
    base_path      : Path = Path('./data/')
    num_workers    : int = 8
    pin_memory     : bool = False


class Dataset(TorchDataset):

    def __init__(
        self,
        base_path,
        files,
        labels=None,
        frame_length=32000,
        frame_sampling=FrameSamplingEnum.DEFAULT,
        num_frames=1,
        augmentation_config=None,
        max_samples=None
    ):
        super().__init__()
        
        self.base_path = base_path
        self.files = files
        self.labels = labels
        self.frame_length = frame_length
        self.frame_sampling = frame_sampling
        self.num_frames = num_frames
        self.max_samples = max_samples
        self.augmentation_config = augmentation_config

        self.augmentation = None
        if augmentation_config and augmentation_config.enable:
            self.augmentation = DataAugmentation(
                augmentation_config,
                self.base_path
            )

    def __len__(self):
        if self.max_samples:
            return min(len(self.files), self.max_samples)
        return len(self.files)

    def preprocess_data(self, data, augment=True):
        if augment and self.augmentation:
            assert data.ndim == 2 and data.shape[0] == 1 # (1, T)
            data = self.augmentation(data)        
        return data

    def __getitem__(self, i):
        data = load_audio(
            self.base_path / self.files[i],
            frame_length=self.frame_length,
            num_frames=self.num_frames
        ) # (N, T)

        x = torch.FloatTensor(self.preprocess_data(data)).squeeze(0)

        info = {'files': self.files[i]}
        if self.labels:
            info.update({'labels': self.labels[i]})

        return i, x, info
