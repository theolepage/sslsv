from typing import Optional, Tuple, Union

from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio(path: Union[Path, str]) -> Tuple[np.ndarray, int]:
    """
    Read audio data from a file.

    Args:
        path (Union[Path, str]): Path to the audio file.

    Returns:
        Tuple[np.ndarray, int]: Audio data and sample rate.
    """
    return sf.read(path)


def load_audio(
    path: Union[Path, str],
    frame_length: int,
    num_frames: int = 1,
    min_length: Optional[int] = None,
) -> np.ndarray:
    """
    Load audio data from a file and extract frames of specified length.

    Args:
        path (Union[Path, str]): Path to the audio file.
        frame_length (int): Frame length.
        num_frames (int): Number of frames to sample. Defaults to 1.
        min_length (Optional[int]): Minimum length of the audio data.
            If audio is shorter, it will be padded. Defaults to None.

    Returns:
        np.ndarray: Audio frames. Shape: (N, L).
    """
    audio, sr = read_audio(path)

    # Convert to mono if audio is stereo
    if len(audio.shape) == 2:
        audio = audio.mean(axis=-1)

    # Pad signal if it is shorter than min_length
    if min_length is None:
        min_length = frame_length
    if min_length and len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio) + 1), "wrap")

    # Load entire audio data if frame_length is not specified
    if frame_length is None:
        frame_length = len(audio)

    # Determine frames start indices
    idx = []
    if num_frames == 1:
        idx = [np.random.randint(0, len(audio) - frame_length + 1)]
    else:
        idx = np.linspace(0, len(audio) - frame_length, num=num_frames)

    # Extract frames
    data = [audio[int(i) : int(i) + frame_length] for i in idx]
    data = np.stack(data, axis=0).astype(np.float32)

    return data  # (num_frames, T)
