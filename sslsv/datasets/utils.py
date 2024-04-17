from typing import Optional, Tuple

import numpy as np
import soundfile as sf


def read_audio(path: str) -> Tuple[np.ndarray, int]:
    return sf.read(path)


def load_audio(
    path: str,
    frame_length: int,
    num_frames: int = 1,
    min_length: Optional[int] = None,
) -> np.ndarray:
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
