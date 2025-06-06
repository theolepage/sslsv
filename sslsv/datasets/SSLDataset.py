from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import torch

from sslsv.datasets.Dataset import Dataset, DatasetConfig, FrameSamplingEnum
from sslsv.datasets.utils import load_audio


def sample_ssps_frame(
    audio: np.ndarray,
    frame_length: int,
) -> np.ndarray:
    audio_length = audio.shape[1]

    pos = np.random.randint(0, audio_length - frame_length + 1)
    frame = audio[:, pos : pos + frame_length]

    return frame


def sample_frames(
    audio: np.ndarray,
    frame_length: int,
) -> List[np.ndarray]:
    """
    Sample two frames from an audio signal (default siamese sampling).

    Args:
        audio (np.ndarray): Input audio signal. Shape: (C, L).
        frame_length (int): Length of the frames to sample.

    Returns:
        List[np.ndarray]: List of audio frames.
    """
    audio_length = audio.shape[1]

    pos = np.random.randint(0, audio_length - frame_length + 1)
    frame1 = audio[:, pos : pos + frame_length]

    pos = np.random.randint(0, audio_length - frame_length + 1)
    frame2 = audio[:, pos : pos + frame_length]

    return [frame1, frame2]


def sample_frames_dino(
    audio: np.ndarray,
    global_frames_count: int = 2,
    global_frames_length: int = 4 * 16000,
    local_frames_count: int = 4,
    local_frames_length: int = 2 * 16000,
) -> List[np.ndarray]:
    """
    Sample frames from an audio signal (DINO multi-crops sampling).

    Args:
        audio (np.ndarray): Input audio signal. Shape: (C, L).
        global_frames_count (int): Number of global frames. Defaults to 2.
        global_frames_length (int): Length of global frames. Defaults to 4*16000.
        local_frames_count (int): Number of local frames. Defaults to 4.
        local_frames_length (int): Length of local frames. Defaults to 2*16000.

    Returns:
        List[np.ndarray]: List of audio frames.
    """
    audio_length = audio.shape[1]

    frames = []

    for _ in range(global_frames_count):
        pos = np.random.randint(0, audio_length - global_frames_length + 1)
        frame = audio[:, pos : pos + global_frames_length]
        frames.append(frame)

    for _ in range(local_frames_count):
        pos = np.random.randint(0, audio_length - local_frames_length + 1)
        frame = audio[:, pos : pos + local_frames_length]
        frames.append(frame)

    return frames


class SSLDataset(Dataset):
    """
    PyTorch dataset generating audio frames for self-supervised training.

    Attributes:
        MIN_LOAD_AUDIO_LENGTH (int): Minimum length when loading audio data.
    """

    def __init__(
        self,
        config: DatasetConfig,
        files: List[str],
        labels: Optional[List[int]] = None,
        num_frames: int = 1,
    ):
        """
        Initialize an SSLDataset object.

        Args:
            config (DatasetConfig): Dataset configuration.
            files (List[str]): List of audio file paths.
            labels (Optional[List[int]]): List of labels. Defaults to None.
            num_frames (int): Number of frames to sample. Defaults to 1.
        """
        super().__init__(config, files, labels, num_frames)

        self.MIN_LOAD_AUDIO_LENGTH = max(self.config.frame_length, max(
            self.config.ssl_dino_global_length,
            self.config.ssl_dino_local_length,
        ))

        if self.config.ssps:
            self.MIN_LOAD_AUDIO_LENGTH = max(
                self.MIN_LOAD_AUDIO_LENGTH, self.config.ssps_frame_length
            )

    def _pad_smaller_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Pad smaller frames in a list of audio frames to match the maximum frame length.

        Args:
            frames (List[np.ndarray]): List of audio frames.

        Returns:
            List[np.ndarray]: List of audio frames with equal lengths.
        """
        max_frame_length = max([f.shape[1] for f in frames])
        res = []
        for frame in frames:
            res.append(
                np.concatenate(
                    (
                        frame,
                        np.full((1, max_frame_length - frame.shape[1]), -100),
                    ),
                    axis=-1,
                )
            )
        return res

    def __getitem__(
        self,
        i: Union[int, Tuple[int]],
    ) -> Tuple[int, torch.Tensor, Dict[str, Union[str, int]]]:
        """
        Get multiple audio frames from the dataset.

        Args:
            i (Union[int, Tuple[int]]): Index of the sample.

        Returns:
            Tuple[int, torch.Tensor, Dict[str, Union[str, int]]]: Index, audio data,
                and additional information (file and label).
        """
        if isinstance(i, int):
            file, label = self.files[i], self.labels[i]
            data = load_audio(
                self.config.base_path / file,
                frame_length=None,
                min_length=self.MIN_LOAD_AUDIO_LENGTH,
            )  # (1, T)
            if self.config.frame_sampling == FrameSamplingEnum.DINO:
                frames = sample_frames_dino(
                    data,
                    self.config.ssl_dino_global_count,
                    self.config.ssl_dino_global_length,
                    self.config.ssl_dino_local_count,
                    self.config.ssl_dino_local_length
                )
            else:
                frames = sample_frames(data, self.config.frame_length)
        else:  # isinstance(i, tuple)
            file, label = self.files[i[0]], self.labels[i[0]]
            if self.config.frame_sampling == FrameSamplingEnum.DINO:
                data1 = load_audio(
                    self.config.base_path / self.files[i[0]],
                    frame_length=None,
                    min_length=self.MIN_LOAD_AUDIO_LENGTH,
                )
                data2 = load_audio(
                    self.config.base_path / self.files[i[1]],
                    frame_length=None,
                    min_length=self.MIN_LOAD_AUDIO_LENGTH,
                )
                frames1 = sample_frames_dino(
                    data1,
                    self.config.ssl_dino_global_count,
                    self.config.ssl_dino_global_length,
                    self.config.ssl_dino_local_count,
                    self.config.ssl_dino_local_length
                )
                frames2 = sample_frames_dino(
                    data2,
                    self.config.ssl_dino_global_count,
                    self.config.ssl_dino_global_length,
                    self.config.ssl_dino_local_count,
                    self.config.ssl_dino_local_length
                )
                frames = [frames1[i] if i < 2 else frames2[i] for i in range(len(frames1))]
            else:
                frame1 = load_audio(
                    self.config.base_path / self.files[i[0]],
                    frame_length=self.config.frame_length,
                )
                frame2 = load_audio(
                    self.config.base_path / self.files[i[1]],
                    frame_length=self.config.frame_length,
                )
                frames = [frame1, frame2]

        frames = [self.preprocess_data(f) for f in frames]

        if self.config.ssps:
            frames.append(sample_ssps_frame(data, self.config.ssps_frame_length))

        frames = self._pad_smaller_frames(frames)

        x = torch.FloatTensor(np.concatenate(frames, axis=0))

        info = {"files": file}
        if self.labels:
            info.update({"labels": label})

        return i, x, info
