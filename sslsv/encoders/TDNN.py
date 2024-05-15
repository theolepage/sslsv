from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


@dataclass
class TDNNConfig(BaseEncoderConfig):
    """
    TDNN encoder configuration.

    Attributes:
        nb_blocks (int): Number of TDNN blocks.
        channels (List[int]): Number of channels for each TDNN block.
        kernel_sizes (List[int]): Kernel sizes for each TDNN block.
        dilations (List[int]): Dilations for each TDNN block.
    """

    nb_blocks: int = 5

    channels: List[int] = field(default_factory=lambda: [512, 512, 512, 512, 1500])

    kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 1, 1])

    dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 1, 1])


class TDNNBlock(nn.Module):
    """
    TDNN main module.

    Attributes:
        conv (nn.Conv1d): Convolutional layer.
        activation (nn.LeakyReLU): Activation function.
        bn (nn.BatchNorm1d): Batch normalization layer.
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation: int):
        """
        Initialize a TDNNBlock module.

        Args:
            in_dim (int): Number of input channels.
            out_dim (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation (int): Dilation factor for the convolution.

        Returns:
            None
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_dim, out_dim, kernel_size=kernel_size, dilation=dilation
        )
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.bn(self.activation(self.conv(X)))


class TDNN(BaseEncoder):
    """
    Time Delay Neural Network (TDNN) encoder.

    Paper:
        X-vectors: Robust dnn embeddings for speaker recognition
        *David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, Sanjeev Khudanpur*
        ICASSP 2018
        https://www.danielpovey.com/files/2018_icassp_xvectors.pdf

    Attributes:
        blocks (nn.Sequential): Sequential module of TDNN blocks.
        last_fc (nn.Linear): Final linear layer.
    """

    def __init__(self, config: TDNNConfig):
        """
        Initialize a TDNN encoder.

        Args:
            config (TDNNConfig): Encoder configuration.

        Returns:
            None
        """
        super().__init__(config)

        self.blocks = nn.Sequential(
            *[
                TDNNBlock(
                    in_dim=config.channels[i - 1] if i > 0 else config.mel_n_mels,
                    out_dim=config.channels[i],
                    kernel_size=config.kernel_sizes[i],
                    dilation=config.dilations[i],
                )
                for i in range(config.nb_blocks)
            ]
        )

        self.last_fc = nn.Linear(config.channels[-1] * 2, config.encoder_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        Z = super().forward(X)
        # Z shape: (B, C, L) = (B, 40, 200)

        Z = self.blocks(Z)

        # Stats pooling
        mean = torch.mean(Z, dim=-1)
        std = torch.std(Z, dim=-1)
        stats = torch.cat((mean, std), dim=1)

        Z = self.last_fc(stats)

        return Z
