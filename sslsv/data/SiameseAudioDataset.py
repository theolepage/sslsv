import numpy as np

import torch

from sslsv.data.AudioDataset import AudioDataset
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


class SiameseAudioDataset(AudioDataset):

    FRAME_SAMPLING_METHODS = {
        'default': sample_frames,
        'dino':    sample_frames_dino
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            min_length = 2 * self.frame_length
            if self.frame_sampling == 'dino': min_length = self.frame_length

            data = load_audio(
                self.base_path / self.files[i],
                frame_length=None,
                min_length=min_length
            ) # (1, T)
            frames = self.FRAME_SAMPLING_METHODS[self.frame_sampling](
                data,
                self.frame_length
            )
            y = self.labels[i]
        else:
            frame1 = load_audio(
                self.base_path / self.files[i[0]],
                frame_length=self.frame_length
            )
            frame2 = load_audio(
                self.base_path / self.files[i[1]],
                frame_length=self.frame_length
            )
            frames = [frame1, frame2]
            y = self.labels[i[0]]

        frames = [self.preprocess_data(f) for f in frames]
        
        frames = self._pad_smaller_frames(frames)
        
        x = torch.FloatTensor(np.concatenate(frames, axis=0))

        info = {'files': self.files[i]}
        if self.labels:
            info.update({'labels': self.labels[i]})

        return i, x, info
