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

    spec_aug: bool = False
    spec_aug_prob: float = 0.2
    spec_aug_max_time: int = 5
    spec_aug_max_freq: int = 3


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

        self.spec_aug = config.spec_aug
        self.spec_aug_prob = config.spec_aug_prob
        self.spec_aug_max_time = config.spec_aug_max_time
        self.spec_aug_max_freq = config.spec_aug_max_freq

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

    def _apply_spec_aug(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment.

        Args:
            Z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        N, D, T = Z.shape

        apply_mask = torch.rand(N) < self.spec_aug_prob  # (N,)

        # Time masking
        t_widths = torch.randint(0, self.spec_aug_max_time + 1, (N,))
        t_starts = torch.randint(0, T - self.spec_aug_max_time, (N,))
        t_idx = torch.arange(T).unsqueeze(0)  # (1, T)
        t_mask = (t_idx >= t_starts.unsqueeze(1)) & (t_idx < (t_starts + t_widths).unsqueeze(1))  # (N, T)
        t_mask = t_mask & apply_mask.unsqueeze(1)

        # Frequency masking
        f_widths = torch.randint(0, self.spec_aug_max_freq + 1, (N,))
        f_starts = torch.randint(0, D - self.spec_aug_max_freq, (N,))
        f_idx = torch.arange(D).unsqueeze(0)  # (1, D)
        f_mask = (f_idx >= f_starts.unsqueeze(1)) & (f_idx < (f_starts + f_widths).unsqueeze(1))  # (N, D)
        f_mask = f_mask & apply_mask.unsqueeze(1)

        # Apply masks
        Z[t_mask.unsqueeze(1).expand_as(Z)] = 0
        Z[f_mask.unsqueeze(2).expand_as(Z)] = 0

        return Z

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
                Z = self.features_extractor(X)
                if self.spec_aug and self.training and X.size(-1) == 32000:
                    Z = self._apply_spec_aug(Z)
                Z = Z + 1e-6
                Z = Z.log()
                Z = self.instance_norm(Z)
            # Z: (B, C, L) = (B, 40, 200)
        else:
            Z = X

        return Z
