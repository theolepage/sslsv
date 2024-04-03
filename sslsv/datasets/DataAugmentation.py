from dataclasses import dataclass
from typing import Tuple

import glob
import os
import numpy as np
import random
from scipy.signal import convolve

from sslsv.datasets.utils import load_audio, read_audio


@dataclass
class DataAugmentationConfig:

    enable: bool = True
    rir: bool = True
    musan: bool = True
    musan_nb_iters: int = 1
    musan_noise_snr:  Tuple[int, int] = (0, 15)
    musan_speech_snr: Tuple[int, int] = (13, 20)
    musan_music_snr:  Tuple[int, int] = (5, 15)
    musan_noise_num:  Tuple[int, int] = (1, 1)
    musan_speech_num: Tuple[int, int] = (1, 1) # (3, 7)
    musan_music_num:  Tuple[int, int] = (1, 1)


class DataAugmentation:

    def __init__(self, config, base_path):
        self.config = config

        self.rir_path = base_path / 'simulated_rirs' / '*/*/*.wav'
        self.rir_files = glob.glob(str(self.rir_path))

        self.musan_files = {}
        self.musan_path = base_path / 'musan_split' / '*/*/*.wav'
        for file in glob.glob(str(self.musan_path)):
            category = file.split(os.sep)[-3]
            if not category in self.musan_files:
                self.musan_files[category] = []
            self.musan_files[category].append(file)

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)

        rir, fs = read_audio(rir_file)
        rir = rir.reshape((1, -1)).astype(np.float32)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        
        return convolve(audio, rir, mode='full')[:, :audio.shape[1]]

    def _get_noise_snr(self, category):
        CATEGORY_TO_SNR = {
            'noise'  : self.config.musan_noise_snr,
            'speech' : self.config.musan_speech_snr,
            'music'  : self.config.musan_music_snr
        }
        min_, max_ = CATEGORY_TO_SNR[category]
        return random.uniform(min_, max_)

    def _get_noise_num(self, category):
        CATEGORY_TO_NUM = {
            'noise'  : self.config.musan_noise_num,
            'speech' : self.config.musan_speech_num,
            'music'  : self.config.musan_music_num
        }
        min_, max_ = CATEGORY_TO_NUM[category]
        return random.randint(min_, max_)

    def add_noise(self, audio, category):
        noise_files = random.sample(
            self.musan_files[category],
            self._get_noise_num(category)
        )

        noises = []
        for noise_file in noise_files:
            noise = load_audio(noise_file, audio.shape[1])
            
            # Determine noise scale factor according to desired SNR
            clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
            noise_db = 10 * np.log10(np.mean(noise[0] ** 2) + 1e-4)
            noise_snr = self._get_noise_snr(category)
            noise_scale = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10))

            noises.append(noise * noise_scale)

        noises = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noises + audio

    def __call__(self, audio):
        if self.config.musan:
            for _ in range(self.config.musan_nb_iters):
                musan_category = random.randint(0, 2)
                if musan_category == 0:
                    audio = self.add_noise(audio, 'music')
                elif musan_category == 1:
                    audio = self.add_noise(audio, 'speech')
                elif musan_category == 2:
                    audio = self.add_noise(audio, 'noise')
        if self.config.rir:
            audio = self.reverberate(audio)
        return audio