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
    frame_length: int,
    large_frames_count: int = 2,
    large_frames_length: int = 4 * 16000,
    small_frames_count: int = 4,
    small_frames_length: int = 2 * 16000,
) -> List[np.ndarray]:
    """
    Sample frames from an audio signal (DINO multi-crops sampling).

    Args:
        audio (np.ndarray): Input audio signal. Shape: (C, L).
        frame_length (int): Unused.
        large_frames_count (int): Number of large frames. Defaults to 2.
        large_frames_length (int): Length of large frames. Defaults to 4*16000.
        small_frames_count (int): Number of small frames. Defaults to 4.
        small_frames_length (int): Length of small frames. Defaults to 2*16000.

    Returns:
        List[np.ndarray]: List of audio frames.
    """
    audio_length = audio.shape[1]

    frames = []

    for _ in range(large_frames_count):
        pos = np.random.randint(0, audio_length - large_frames_length + 1)
        frame = audio[:, pos : pos + large_frames_length]
        frames.append(frame)

    for _ in range(small_frames_count):
        pos = np.random.randint(0, audio_length - small_frames_length + 1)
        frame = audio[:, pos : pos + small_frames_length]
        frames.append(frame)

    return frames


class SSLDataset(Dataset):
    """
    PyTorch dataset generating audio frames for self-supervised training.

    Attributes:
        MIN_LOAD_AUDIO_LENGTH (int): Minimum length when loading audio data.
        FRAME_SAMPLING_METHODS (Dict[FrameSamplingEnum, Callable[..., List[np.ndarray]]]): Dictionary
            mapping frame sampling options to corresponding methods.
    """

    FRAME_SAMPLING_METHODS = {
        FrameSamplingEnum.DEFAULT: sample_frames,
        FrameSamplingEnum.DINO: sample_frames_dino,
    }

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

        self.MIN_LOAD_AUDIO_LENGTH = 64000

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
            frames = self.FRAME_SAMPLING_METHODS[self.config.frame_sampling](
                data, self.config.frame_length
            )
        else:  # isinstance(i, tuple)
            file, label = self.files[i[0]], self.labels[i[0]]
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
