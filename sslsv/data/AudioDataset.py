import os
import numpy as np

import torch
from torch.utils.data import Dataset

from sslsv.data.AudioAugmentation import AudioAugmentation
from sslsv.data.utils import load_audio


class AudioDataset(Dataset):

    def __init__(self, config):
        self.config = config

        # Create augmentation module
        self.wav_augment = None
        if self.config.wav_augment.enable:
            self.wav_augment = AudioAugmentation(
                self.config.wav_augment,
                self.config.base_path
            )

        self.load_data()

    def load_data(self):
        # Create lists of audio paths and labels
        self.files = []
        self.labels = []
        self.nb_classes = 0
        labels_id = {}
        for line in open(self.config.base_path / self.config.train):
            label, file = line.rstrip().split()

            self.files.append(self.config.base_path / file)

            if label not in labels_id:
                labels_id[label] = self.nb_classes
                self.nb_classes += 1
            self.labels.append(labels_id[label])

    def __len__(self):
        if self.config.max_samples: return self.config.max_samples
        return len(self.labels)

    def preprocess_data(self, data, augment=True):
        assert data.ndim == 2 and data.shape[0] == 1 # (1, T)
        if augment and self.wav_augment: data = self.wav_augment(data)        
        return data

    def __getitem__(self, i):
        data = load_audio(
            self.files[i],
            frame_length=self.config.frame_length
        ) # (1, T)

        x = torch.FloatTensor(self.preprocess_data(data)).squeeze(0)

        y = self.labels[i]

        return i, x, y
