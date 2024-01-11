import os
import numpy as np

import torch
from torch.utils.data import Dataset

from sslsv.data.AudioAugmentation import AudioAugmentation
from sslsv.data.utils import load_audio


class AudioDataset(Dataset):

    def __init__(
        self,
        base_path,
        files,
        labels=None,
        frame_length=32000,
        frame_sampling='default',
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

        # Create augmentation module
        self.augmentation = None
        if augmentation_config and augmentation_config.enable:
            self.augmentation = AudioAugmentation(
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
