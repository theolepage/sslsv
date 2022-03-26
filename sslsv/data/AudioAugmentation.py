import glob
import os
import numpy as np
import random
from scipy.signal import convolve
import soundfile as sf

from sslsv.data.utils import load_audio, read_audio


class AudioAugmentation:

    def __init__(self, config, base_path):
        self.config = config

        self.rir_path = os.path.join(base_path, 'simulated_rirs', '*/*/*.wav')
        self.rir_files = glob.glob(self.rir_path)

        self.musan_files = {}
        self.musan_path = os.path.join(base_path, 'musan_split', '*/*/*.wav')
        for file in glob.glob(self.musan_path):
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

    def get_noise_snr(self, category):
        min_, max_ = self.config.musan_noise_snr # category == 'noise'
        if category == 'speech':
            min_, max_ = self.config.musan_speech_snr
        elif category == 'music':
            min_, max_ = self.config.musan_music_snr
        return random.uniform(min_, max_)

    def add_noise(self, audio, category):
        noise_file = random.choice(self.musan_files[category])
        noise = load_audio(noise_file, audio.shape[1])
        
        # Determine noise scale factor according to desired SNR
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4) 
        noise_db = 10 * np.log10(np.mean(noise[0] ** 2) + 1e-4) 
        noise_snr = self.get_noise_snr(category)
        noise_scale = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10))

        return noise * noise_scale + audio

    def __call__(self, audio):
        if self.config.musan:
            transform_type = random.randint(0, 2)
            if transform_type == 0:
                audio = self.add_noise(audio, 'music')
            elif transform_type == 1:
                audio = self.add_noise(audio, 'speech')
            elif transform_type == 2:
                audio = self.add_noise(audio, 'noise')
        if self.config.rir:
            audio = self.reverberate(audio)
        return audio