import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from dataclasses import dataclass

from sslsv.configs import EncoderConfig


class AudioPreEmphasis(nn.Module):

    def __init__(self, coeff=0.97):
        super().__init__()

        self.w = torch.FloatTensor([-coeff, 1.0]).unsqueeze(0).unsqueeze(0)

    def forward(self, audio):
        audio = audio.unsqueeze(1)
        audio = F.pad(audio, (1, 0), 'reflect')
        return F.conv1d(audio, self.w.to(audio.device)).squeeze(1)


@dataclass
class BaseEncoderConfig(EncoderConfig):

    encoder_dim: int = 512

    extract_mel_features: bool = True
    mel_n_mels = 40
    mel_n_fft = 512
    mel_win_length = 400 # 25ms
    mel_hop_length = 160 # 10ms


class BaseEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder_dim = config.encoder_dim

        if config.extract_mel_features:
            self.features_extractor = nn.Sequential(
                AudioPreEmphasis(),
                MelSpectrogram(
                    n_fft=config.mel_n_fft,
                    win_length=config.mel_win_length,
                    hop_length=config.mel_hop_length,
                    window_fn=torch.hamming_window,
                    n_mels=config.mel_n_mels
                )
            )
            self.instance_norm = nn.InstanceNorm1d(config.mel_n_mels)

    def forward(self, X):
        # X: (B, 32000)

        if self.features_extractor:
            with torch.no_grad():
                Z = self.features_extractor(X) + 1e-6
                Z = Z.log()
                Z = self.instance_norm(Z)
            # Z: (B, C, L) = (B, 40, 200)

        return Z