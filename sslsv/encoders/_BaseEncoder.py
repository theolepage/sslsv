import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

from dataclasses import dataclass


@dataclass
class BaseEncoderConfig:
    """
    Base configuration for encoders.

    Attributes:
        encoder_dim (int): Output dimension of the encoder.
        extract_mel_features (bool): Whether to extract Mel Spectrogram features.
        mel_n_mels (int): number of mel filterbanks for Mel Spectrogram.
        mel_n_fft (int): FFT size for Mel Spectrogram.
        mel_win_fn (str): Window function for Mel Spectrogram.
        mel_win_length (int): Window size for Mel Spectrogram.
        mel_hop_length (int): Length of hop between STFT windows for Mel Spectrogram.
        mel_sample_rate (int): Sample rate of audio signal for Mel Spectrogram.
    """

    encoder_dim: int = 512

    extract_mel_features: bool = True
    mel_n_mels: int = 40
    mel_n_fft: int = 512
    mel_win_fn: str = "hamming"
    mel_win_length: int = 400  # 25ms
    mel_hop_length: int = 160  # 10ms
    mel_sample_rate: int = 16000  # 16kHz


class BaseEncoder(nn.Module):
    """
    Base class for encoders.

    Attributes:
        MEL_WIN_FUNCTIONS (Dict[str, Callable[..., torch.Tensor]]): Dictionary mapping
            window function names to corresponding functions.

        encoder_dim (int): Output dimension of the encoder.
        features_extractor (nn.Module): Feature extraction module.
        instance_norm (nn.Module): Feature normalization module.
    """

    MEL_WIN_FUNCTIONS = {
        "hamming": torch.hamming_window,
        "hann": torch.hann_window,
    }

    def __init__(self, config: BaseEncoderConfig):
        """
        Initialize a base encoder.

        Args:
            config (BaseEncoderConfig): Encoder configuration.

        Returns:
            None
        """
        super().__init__()

        self.encoder_dim = config.encoder_dim

        self.features_extractor = None
        if config.extract_mel_features:
            self.features_extractor = nn.Sequential(
                MelSpectrogram(
                    n_fft=config.mel_n_fft,
                    win_length=config.mel_win_length,
                    hop_length=config.mel_hop_length,
                    window_fn=self.MEL_WIN_FUNCTIONS[config.mel_win_fn],
                    n_mels=config.mel_n_mels,
                    sample_rate=config.mel_sample_rate,
                )
            )
            self.instance_norm = nn.InstanceNorm1d(config.mel_n_mels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, L).

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.features_extractor:
            with torch.no_grad():
                Z = self.features_extractor(X) + 1e-6
                Z = Z.log()
                Z = self.instance_norm(Z)
            # Z: (B, C, L) = (B, 40, 200)
        else:
            Z = X

        return Z
