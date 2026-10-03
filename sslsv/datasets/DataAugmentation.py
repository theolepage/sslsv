from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

from pathlib import Path
import glob
import os
import numpy as np
import random
from scipy.signal import convolve

from sslsv.datasets.utils import load_audio, read_audio


class DataAugmentationStrategyEnum(Enum):
    """
    Enumeration representing data-augmentation strategies.

    Attributes:
        REVERB (str): Apply reverberation.
        NOISE (str): Apply noise.
        EITHER (str): Apply reverberation or noise.
        BOTH (str): Apply reverberation and noise.
        ALL (str): Apply reverberation or noise or both or nothing.
    """

    REVERB = "reverb"
    NOISE = "noise"
    EITHER = "reverb|noise"
    BOTH = "reverb+noise"
    ALL = "all"


@dataclass
class DataAugmentationConfig:
    """
    Data-augmentation configuration.

    Attributes:
        enable (bool): Whether data-augmentation is enabled.
        aug_prob (float): Probability of applying augmentations
        strategy (DataAugmentationStrategyEnum): Data-augmentation techniques to apply.
        strategy_probs (List[float]): Probabilities of applying nothing, reverberation,
            noise and both, used by the ALL strategy. Defaults to None, which draws
            the four uniformly.
        musan_noise_snr (Tuple[int, int]): Signal-to-noise ratio (SNR) range for MUSAN noise.
        musan_speech_snr (Tuple[int, int]): Signal-to-noise ratio (SNR) range for MUSAN speech.
        musan_music_snr (Tuple[int, int]): Signal-to-noise ratio (SNR) range for MUSAN music.
        musan_noise_num (Tuple[int, int]): Range for selecting the number of MUSAN noises to apply.
        musan_speech_num (Tuple[int, int]): Range for selecting the number of MUSAN speeches to apply.
        musan_music_num (Tuple[int, int]): Range for selecting the number of MUSAN musics to apply.
        musan_file (str): Path, relative to the base path, of a Kaldi-style `wav.scp`
            listing the MUSAN files to use instead of globbing `musan_split`. Read the
            way 3D-Speaker reads it, keyed by the first column, so repeated keys keep
            only their last path.
        rir_file (str): Path, relative to the base path, of a `.npy` bank of RIRs to
            use instead of the `simulated_rirs` wav files (SDPN uses `rir.npy`).
        rir_normalize (bool): Whether to scale RIRs to unit energy before convolving.
        rir_gain (List[int]): Range of the random gain applied to the RIR, as
            `10 ** (0.1 * gain)`. Defaults to None, which applies no gain.
        snr_eps (float): Floor added to the signal and noise powers before converting
            them to decibels. SDPN adds `1e-4` to audio read as int16, where it is
            negligible; on audio read as float in [-1, 1] the equivalent floor is
            `1e-4 / 32768 ** 2`.
    """

    enable: bool = True

    aug_prob: float = 1.0

    strategy: DataAugmentationStrategyEnum = DataAugmentationStrategyEnum.BOTH
    strategy_probs: Optional[List[float]] = None

    musan_noise_snr: Tuple[int, int] = (0, 15)
    musan_speech_snr: Tuple[int, int] = (13, 20)
    musan_music_snr: Tuple[int, int] = (5, 15)
    musan_noise_num: Tuple[int, int] = (1, 1)
    musan_speech_num: Tuple[int, int] = (1, 1)  # (3, 7)
    musan_music_num: Tuple[int, int] = (1, 1)

    musan_file: Optional[str] = None
    rir_file: Optional[str] = None
    rir_normalize: bool = True
    rir_gain: Optional[List[int]] = None

    snr_eps: float = 1e-4


class DataAugmentation:
    """
    Apply data-augmentation (noise and reverberation) to audio signals.

    Attributes:
        config (DataAugmentationConfig): Data-augmentation configuration.
        base_path (Path): Base path for all files.
        rir_files (List[str]): List of RIR file paths.
        musan_files (Dict[str, List[str]]): Dictionary mapping MUSAN categories to lists of noise files.
    """

    def __init__(self, config: DataAugmentationConfig, base_path: Path):
        """
        Initialize a DataAugmentation object.

        Args:
            config (DataAugmentationConfig): Data-augmentation configuration.
            base_path (Path): Base path for all files.

        Returns:
            None
        """
        self.config = config

        # SDPN draws its RIRs from a fixed `.npy` bank instead of the wav files.
        self.rirs = None
        self.rir_files = []
        if config.rir_file:
            self.rirs = np.load(str(base_path / config.rir_file))
        else:
            rir_path = base_path / "simulated_rirs" / "*/*/*.wav"
            self.rir_files = glob.glob(str(rir_path))

        self.musan_files = {}
        if config.musan_file:
            for file in self._load_wav_scp(base_path / config.musan_file):
                # 3D-Speaker reads the category from the fourth component from the
                # end, its noise files being laid out as
                # `musan_split/<category>/<source>/<recording>/<segment>.wav`.
                category = file.split("/")[-4]
                self.musan_files.setdefault(category, []).append(file)
            return

        musan_path = base_path / "musan_split" / "*/*/*.wav"
        for file in glob.glob(str(musan_path)):
            category = file.split(os.sep)[-3]
            if not category in self.musan_files:
                self.musan_files[category] = []
            self.musan_files[category].append(file)

    @staticmethod
    def _load_wav_scp(path: Path) -> List[str]:
        """
        Read the file paths of a Kaldi-style `wav.scp`, reproducing 3D-Speaker's
        `load_wav_scp`.

        3D-Speaker builds a dictionary keyed by the first column, so a key appearing
        on several rows only keeps the path of its last row. Its MUSAN list keys every
        segment by the recording it was cut from, which collapses 128924 segments down
        to the 1768 that SDPN actually trains on. Reproducing SDPN requires reproducing
        that, hence the dictionary here.

        Args:
            path (Path): Path to the `wav.scp` file.

        Returns:
            List[str]: File paths.
        """
        files = {}
        with open(path) as f:
            for line in f:
                columns = line.split()
                if len(columns) >= 2:
                    files[columns[0]] = columns[1]
        return list(files.values())

    def reverberate(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply reverberation to an audio signal using a randomly sampled
        Room Impulse Response (RIR).

        Args:
            audio (np.ndarray): Input audio.

        Returns:
            np.ndarray: Output audio.
        """
        if self.rirs is not None:
            rir = random.choice(self.rirs)
        else:
            rir, fs = read_audio(random.choice(self.rir_files))
        rir = rir.reshape((1, -1)).astype(np.float32)

        if self.config.rir_normalize:
            rir = rir / np.sqrt(np.sum(rir**2))

        if self.config.rir_gain:
            gain = np.random.uniform(*self.config.rir_gain)
            rir = np.multiply(rir, pow(10, 0.1 * gain))

        return convolve(audio, rir, mode="full")[:, : audio.shape[1]]

    def _get_noise_snr(self, category: str) -> float:
        """
        Get a random signal-to-noise ratio (SNR) value for a MUSAN category.

        Args:
            category (str): MUSAN category ('noise', 'speech', or 'music').

        Returns:
            float: Random SNR value.
        """
        CATEGORY_TO_SNR = {
            "noise": self.config.musan_noise_snr,
            "speech": self.config.musan_speech_snr,
            "music": self.config.musan_music_snr,
        }
        min_, max_ = CATEGORY_TO_SNR[category]
        return random.uniform(min_, max_)

    def _get_noise_num(self, category: str) -> int:
        """
        Get a random number of noises to add for a MUSAN category.

        Args:
            category (str): MUSAN category ('noise', 'speech', or 'music').

        Returns:
            int: Random number of noises to apply.
        """
        CATEGORY_TO_NUM = {
            "noise": self.config.musan_noise_num,
            "speech": self.config.musan_speech_num,
            "music": self.config.musan_music_num,
        }
        min_, max_ = CATEGORY_TO_NUM[category]
        return random.randint(min_, max_)

    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Add noise to an audio signal using a randomly selected noise from MUSAN.

        Args:
            audio (numpy.ndarray): Input audio.

        Returns:
            numpy.ndarray: Output audio.
        """
        category = random.choice(["speech", "noise", "music"])

        noise_files = random.sample(
            self.musan_files[category], self._get_noise_num(category)
        )

        noises = []
        for noise_file in noise_files:
            noise = load_audio(noise_file, audio.shape[1])

            # Determine noise scale factor according to desired SNR
            eps = self.config.snr_eps
            clean_db = 10 * np.log10(np.mean(audio**2) + eps)
            noise_db = 10 * np.log10(np.mean(noise[0] ** 2) + eps)
            noise_snr = self._get_noise_snr(category)
            noise_scale = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10))

            noises.append(noise * noise_scale)

        noises = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noises + audio

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise and reverberation to the input audio.

        Args:
            audio (np.ndarray): Input audio.

        Returns:
            np.ndarray: Output audio.
        """
        if random.random() >= self.config.aug_prob:
            return audio

        if self.config.strategy == DataAugmentationStrategyEnum.REVERB:
            audio = self.reverberate(audio)
        elif self.config.strategy == DataAugmentationStrategyEnum.NOISE:
            audio = self.add_noise(audio)
        elif self.config.strategy == DataAugmentationStrategyEnum.EITHER:
            aug_type = random.randint(1, 2)
            if aug_type == 1:
                audio = self.reverberate(audio)
            else:
                audio = self.add_noise(audio)
        elif self.config.strategy == DataAugmentationStrategyEnum.BOTH:
            audio = self.reverberate(audio)
            audio = self.add_noise(audio)
        elif self.config.strategy == DataAugmentationStrategyEnum.ALL:
            if self.config.strategy_probs:
                aug_type = random.choices(
                    range(4), weights=self.config.strategy_probs
                )[0]
            else:
                aug_type = random.randint(0, 3)
            if aug_type == 0:
                pass
            elif aug_type == 1:
                audio = self.reverberate(audio)
            elif aug_type == 2:
                audio = self.add_noise(audio)
            elif aug_type == 3:
                audio = self.reverberate(audio)
                audio = self.add_noise(audio)

        return audio
