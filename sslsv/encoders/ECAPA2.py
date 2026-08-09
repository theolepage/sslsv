from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


class SEBlock2d(nn.Module):
    """
    Squeeze-and-Excitation (SE) module for 2D feature maps.

    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling module.
        fc (nn.Sequential): Bottleneck fully-connected module.
    """

    def __init__(self, channels: int, se_channels: int = 128):
        """
        Initialize a SEBlock2d module.

        Args:
            channels (int): Number of input channels.
            se_channels (int): Number of SE channels. Defaults to 128.

        Returns:
            None
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, se_channels),
            nn.ReLU(),
            nn.BatchNorm1d(se_channels),
            nn.Linear(se_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, H, W).
        """
        N, C, _, _ = X.size()

        Z = self.avg_pool(X).view(N, C)
        Z = self.fc(Z).view(N, C, 1, 1)
        return X * Z


class SEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation (SE) module for 1D feature maps.

    Attributes:
        fc (nn.Sequential): Bottleneck fully-connected module.
    """

    def __init__(self, channels: int, se_channels: int = 128):
        """
        Initialize a SEBlock1d module.

        Args:
            channels (int): Number of input channels.
            se_channels (int): Number of SE channels. Defaults to 128.

        Returns:
            None
        """
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(channels, se_channels),
            nn.ReLU(),
            nn.BatchNorm1d(se_channels),
            nn.Linear(se_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, L).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, L).
        """
        Z = self.fc(X.mean(dim=2)).unsqueeze(2)
        return X * Z.expand_as(X)


class LFEBlock(nn.Module):
    """
    Local Feature Extractor (LFE) module.

    Consists of three 2D convolutions followed by a Squeeze-and-Excitation module.
    The residual connection is discarded when the number of channels changes without
    any striding.

    Attributes:
        convs (nn.Sequential): Convolutional module.
        se (SEBlock2d): Squeeze-and-Excitation module.
        shortcut (Optional[nn.Sequential]): Residual connection module.
        residual (bool): Whether to use a residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_channels: int = 128,
        stride: int = 1,
    ):
        """
        Initialize a LFEBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            se_channels (int): Number of SE channels. Defaults to 128.
            stride (int): Stride of the first convolution along the frequency
                dimension. Defaults to 1.

        Returns:
            None
        """
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(stride, 1),
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock2d(out_channels, se_channels)

        self.residual = True
        self.shortcut = None
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        elif in_channels != out_channels:
            self.residual = False

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C', H', W).
        """
        Z = self.se(self.convs(X))

        if not self.residual:
            return Z

        return Z + (self.shortcut(X) if self.shortcut else X)


class Res2NetBlock(nn.Module):
    """
    Res2Net module.

    Attributes:
        scale (int): Scale factor for the number of channels.
        convs (nn.ModuleList): List of convolution modules.
        norms (nn.ModuleList): List of normalization modules.
    """

    def __init__(
        self,
        channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 2,
    ):
        """
        Initialize a Res2NetBlock module.

        Args:
            channels (int): Number of input and output channels.
            scale (int): Scale factor for the number of channels. Defaults to 8.
            kernel_size (int): Size of the convolutional kernel. Defaults to 3.
            dilation (int): Dilation rate of the convolutions. Defaults to 2.

        Raises:
            AssertionError: If the number of channels is not divisible by the scale factor.
        """
        super().__init__()

        assert channels % scale == 0

        self.scale = scale

        hidden_channels = channels // scale

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=dilation * (kernel_size - 1) // 2,
                    padding_mode="reflect",
                )
                for _ in range(scale - 1)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(scale - 1)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, L).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, L).
        """
        Y = []
        for i, X_i in enumerate(torch.chunk(X, self.scale, dim=1)):
            if i == 0:
                Y_i = X_i
            elif i == 1:
                Y_i = self.norms[i - 1](F.relu(self.convs[i - 1](X_i)))
            else:
                Y_i = self.norms[i - 1](F.relu(self.convs[i - 1](X_i + Y_i)))
            Y.append(Y_i)
        return torch.cat(Y, dim=1)


class SERes2NetBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Res2Net module of the Global Feature Extractor (GFE).

    Attributes:
        block (nn.Sequential): Sequence of convolution, Res2Net and
            Squeeze-and-Excitation modules.
    """

    def __init__(
        self,
        channels: int,
        se_channels: int = 128,
        res2net_scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 2,
    ):
        """
        Initialize a SERes2NetBlock module.

        Args:
            channels (int): Number of input and output channels.
            se_channels (int): Number of SE channels. Defaults to 128.
            res2net_scale (int): Scale factor for the number of channels in Res2Net.
                Defaults to 8.
            kernel_size (int): Size of the Res2Net convolutional kernel. Defaults to 3.
            dilation (int): Dilation rate of the Res2Net convolutions. Defaults to 2.

        Returns:
            None
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            Res2NetBlock(channels, res2net_scale, kernel_size, dilation),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            SEBlock1d(channels, se_channels),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, L).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, L).
        """
        return self.block(X) + X


class ChannelDependentStatisticsPooling(nn.Module):
    """
    Channel-dependent Attentive Statistics (CAS) pooling module.

    Attributes:
        attention (nn.Sequential): Attention module.
    """

    def __init__(self, channels: int, attention_channels: int = 128):
        """
        Initialize a ChannelDependentStatisticsPooling module.

        Args:
            channels (int): Number of input channels.
            attention_channels (int): Number of attention channels. Defaults to 128.

        Returns:
            None
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, L).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, 2*C).
        """
        mean = torch.mean(X, dim=2, keepdim=True).expand_as(X)
        std = torch.sqrt(torch.var(X, dim=2, keepdim=True) + 1e-5).expand_as(X)

        attn = self.attention(torch.cat((X, mean, std), dim=1))
        attn = F.softmax(attn, dim=2)

        X_scaled = attn * X
        mean = torch.sum(X_scaled, dim=2)
        std = torch.sqrt(torch.abs(torch.sum(X_scaled * X, dim=2) - mean**2) + 1e-5)

        return torch.cat((mean, std), dim=1)


@dataclass
class ECAPA2Config(BaseEncoderConfig):
    """
    ECAPA2 encoder configuration.

    Attributes:
        encoder_dim (int): Output dimension of the encoder.
        extract_mel_features (bool): Whether to extract Mel Spectrogram features.
        pooling (bool): Whether to apply temporal pooling.
        spec_n_fft (int): FFT size for the log Spectrogram front-end.
        spec_win_length (int): Window size for the log Spectrogram front-end.
        spec_hop_length (int): Length of hop between STFT windows for the log
            Spectrogram front-end.
        lfe_channels (List[List[int]]): Number of channels of each Local Feature
            Extractor (LFE) module, for each stage.
        lfe_strides (List[int]): Stride along the frequency dimension of the first
            module of each Local Feature Extractor (LFE) stage.
        lfe_se_channels (int): Number of SE channels for the LFE modules.
        gfe_channels (int): Number of channels of the Global Feature Extractor (GFE).
        gfe_out_channels (int): Number of output channels of the Global Feature
            Extractor (GFE).
        gfe_se_channels (int): Number of SE channels for the GFE modules.
        res2net_scale (int): Scale factor for the number of channels in the Res2Net module.
        res2net_kernel_size (int): Size of the Res2Net convolutional kernel.
        res2net_dilation (int): Dilation rate of the Res2Net convolutions.
        attention_channels (int): Number of channels for Channel-dependent Attentive
            Statistics (CAS) pooling.
    """

    encoder_dim: int = 192

    extract_mel_features: bool = False

    pooling: bool = True

    spec_n_fft: int = 511
    spec_win_length: int = 400  # 25ms
    spec_hop_length: int = 160  # 10ms

    lfe_channels: List[List[int]] = field(
        default_factory=lambda: [
            [164, 164, 164],
            [164, 164, 164, 164],
            [164, 164, 192, 192],
            [192, 192, 192, 192],
            [192, 192, 192, 192, 192],
        ]
    )

    lfe_strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2, 2])

    lfe_se_channels: int = 128

    gfe_channels: int = 1024

    gfe_out_channels: int = 1536

    gfe_se_channels: int = 128

    res2net_scale: int = 8

    res2net_kernel_size: int = 3

    res2net_dilation: int = 2

    attention_channels: int = 128


class ECAPA2(BaseEncoder):
    """
    ECAPA2 encoder.

    Paper:
        ECAPA2: A Hybrid Neural Network Architecture and Training Strategy for
        Robust Speaker Embeddings
        *Jenthe Thienpondt, Kris Demuynck*
        ASRU 2023
        https://arxiv.org/abs/2401.08342

    Attributes:
        pooling (bool): Whether to apply temporal pooling.
        spectrogram (Optional[nn.Module]): Log Spectrogram front-end module.
        lfe (nn.ModuleList): List of Local Feature Extractor (LFE) modules.
        tdnn1 (nn.Sequential): First module of the Global Feature Extractor (GFE).
        tdnn2 (SERes2NetBlock): Second module of the Global Feature Extractor (GFE).
        dense (nn.Sequential): Last module of the Global Feature Extractor (GFE).
        asp (ChannelDependentStatisticsPooling): CAS pooling module.
        asp_bn (nn.BatchNorm1d): Batch normalization module for CAS pooling.
        fc (nn.Module): Final linear transformation module.
    """

    def __init__(self, config: ECAPA2Config):
        """
        Initialize an ECAPA2 encoder.

        Args:
            config (ECAPA2Config): Encoder configuration.

        Returns:
            None
        """
        super().__init__(config)

        assert len(config.lfe_channels) == len(config.lfe_strides)

        self.pooling = config.pooling

        self.spectrogram = None
        if not config.extract_mel_features:
            self.spectrogram = Spectrogram(
                n_fft=config.spec_n_fft,
                win_length=config.spec_win_length,
                hop_length=config.spec_hop_length,
            )
            nb_freqs = config.spec_n_fft // 2 + 1
        else:
            nb_freqs = config.mel_n_mels

        # Local Feature Extractor
        self.lfe = nn.ModuleList()
        in_channels = 1
        for channels, stride in zip(config.lfe_channels, config.lfe_strides):
            assert nb_freqs % stride == 0
            nb_freqs //= stride

            for i, out_channels in enumerate(channels):
                self.lfe.append(
                    LFEBlock(
                        in_channels,
                        out_channels,
                        se_channels=config.lfe_se_channels,
                        stride=stride if i == 0 else 1,
                    )
                )
                in_channels = out_channels

        # Global Feature Extractor
        self.tdnn1 = nn.Sequential(
            nn.Conv1d(in_channels * nb_freqs, config.gfe_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(config.gfe_channels),
        )
        self.tdnn2 = SERes2NetBlock(
            config.gfe_channels,
            se_channels=config.gfe_se_channels,
            res2net_scale=config.res2net_scale,
            kernel_size=config.res2net_kernel_size,
            dilation=config.res2net_dilation,
        )
        self.dense = nn.Sequential(
            nn.Conv1d(config.gfe_channels, config.gfe_out_channels, kernel_size=1),
            nn.ReLU(),
        )

        self.asp = ChannelDependentStatisticsPooling(
            config.gfe_out_channels,
            attention_channels=config.attention_channels,
        )
        self.asp_bn = nn.BatchNorm1d(config.gfe_out_channels * 2)

        self.fc = (
            nn.Linear(config.gfe_out_channels * 2, self.encoder_dim)
            if self.pooling
            else nn.Conv1d(config.gfe_out_channels, self.encoder_dim, kernel_size=1)
        )

    def _extract_spectrogram_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Extract log Spectrogram features with mean normalization.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, L).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, D, L').
        """
        with torch.no_grad():
            Z = self.spectrogram(X)
            Z = torch.log(Z + 1e-6)
            Z = Z - Z.mean(dim=-1, keepdim=True)
        return Z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, L).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, D) or (N, D, L').
        """
        Z = super().forward(X)

        if self.spectrogram is not None:
            Z = self._extract_spectrogram_features(Z)
        # Z: (N, D, L) = (N, 256, 201)

        Z = Z.unsqueeze(1)
        # Z: (N, C, H, W) = (N, 1, 256, 201)

        for block in self.lfe:
            Z = block(Z)
        # Z: (N, C, H, W) = (N, 192, 16, 201)

        N, C, H, W = Z.size()
        Z = Z.reshape((N, C * H, W))

        Z = self.tdnn1(Z)
        Z = self.tdnn2(Z)
        Z = self.dense(Z)
        # Z: (N, C, L) = (N, 1536, 201)

        if self.pooling:
            Z = self.asp(Z)
            Z = self.asp_bn(Z)

        Z = self.fc(Z)

        return Z
