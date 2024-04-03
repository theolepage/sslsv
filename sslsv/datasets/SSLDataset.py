import numpy as np

import torch

from sslsv.datasets.Dataset import Dataset, FrameSamplingEnum
from sslsv.datasets.utils import load_audio


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

    return [frame1, frame2]


def sample_frames_dino(
    audio,
    frame_length,
    large_frames_count=2,
    large_frames_length=4*16000,
    small_frames_count=4,
    small_frames_length=2*16000
):
    audio_length = audio.shape[1]

    frames = []

    for _ in range(large_frames_count):
        pos = np.random.randint(0, audio_length - large_frames_length + 1)
        frame = audio[:, pos:pos+large_frames_length]
        frames.append(frame)

    for _ in range(small_frames_count):
        pos = np.random.randint(0, audio_length - small_frames_length + 1)
        frame = audio[:, pos:pos+small_frames_length]
        frames.append(frame)

    return frames


class SSLDataset(Dataset):

    FRAME_SAMPLING_METHODS = {
        FrameSamplingEnum.DEFAULT : sample_frames,
        FrameSamplingEnum.DINO    : sample_frames_dino
    }

    def __init__(self, min_load_audio_length=64000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.min_load_audio_length = min_load_audio_length

    def _pad_smaller_frames(self, frames):
        max_frame_length = max([f.shape[1] for f in frames])
        res = []
        for frame in frames:
            res.append(np.concatenate((
                frame,
                np.zeros((1, max_frame_length - frame.shape[1]))
            ), axis=-1))
        return res

    def __getitem__(self, i):
        if isinstance(i, int):
            file, label = self.files[i], self.labels[i]
            data = load_audio(
                self.base_path / file,
                frame_length=None,
                min_length=self.min_load_audio_length
            ) # (1, T)
            frames = self.FRAME_SAMPLING_METHODS[self.frame_sampling](
                data,
                self.frame_length
            )
        else: # isinstance(i, tuple)
            file, label = self.files[i[0]], self.labels[i[0]]
            frame1 = load_audio(
                self.base_path / self.files[i[0]],
                frame_length=self.frame_length
            )
            frame2 = load_audio(
                self.base_path / self.files[i[1]],
                frame_length=self.frame_length
            )
            frames = [frame1, frame2]

        frames = [self.preprocess_data(f) for f in frames]
        
        frames = self._pad_smaller_frames(frames)
        
        x = torch.FloatTensor(np.concatenate(frames, axis=0))

        info = {'files': file}
        if self.labels:
            info.update({'labels': label})

        return i, x, info
