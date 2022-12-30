import os
import numpy as np

import torch
from torch.utils.data import Dataset

from sslsv.data.AudioDataset import AudioDataset
from sslsv.data.AudioAugmentation import AudioAugmentation
from sslsv.data.utils import load_audio


def sample_frames(audio, frame_length):
    audio_length = audio.shape[1]
    assert audio_length >= 2 * frame_length, \
        "audio_length should >= 2 * frame_length"

    dist = audio_length - 2 * frame_length
    dist = np.random.randint(0, dist + 1)

    lower = frame_length + dist // 2
    upper = audio_length - (frame_length + dist // 2)
    pivot = np.random.randint(lower, upper + 1)

    frame1_from = pivot - dist // 2 - frame_length
    frame1_to = pivot - dist // 2
    frame1 = audio[:, frame1_from:frame1_to]

    frame2_from = pivot + dist // 2
    frame2_to = pivot + dist // 2 + frame_length
    frame2 = audio[:, frame2_from:frame2_to]

    return frame1, frame2


class SiameseAudioDataset(AudioDataset):

    def __init__(self, config):
        super().__init__(config)

    def __getitem__(self, i):
        if isinstance(i, int):
            data = load_audio(
                self.files[i],
                frame_length=None,
                min_length=2*self.config.frame_length
            ) # (1, T)
            frame1, frame2 = sample_frames(data, self.config.frame_length)
            y = self.labels[i]
        else:
            frame1 = load_audio(self.files[i[0]], self.config.frame_length)
            frame2 = load_audio(self.files[i[1]], self.config.frame_length)
            y = self.labels[i[0]]

        X = np.concatenate((
            self.preprocess_data(frame1),
            self.preprocess_data(frame2)
        ), axis=0)
        X = torch.FloatTensor(X)

        return i, X, y
