from dataclasses import dataclass

import torch
from torch import nn

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


@dataclass
class SimpleAudioCNNConfig(BaseEncoderConfig):

    extract_mel_features: bool = False


class SimpleAudioCNNBlock(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.ln = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.act(self.ln(self.conv(X)))


class SimpleAudioCNN(BaseEncoder):

    def __init__(self, config: SimpleAudioCNNConfig):
        super().__init__(config)

        nb_filters = [512, 512, 512, 512, self.encoder_dim]
        kernel_sizes = [10, 8, 4, 4, 4]
        strides = [5, 4, 2, 2, 2]
        paddings = [3, 2, 1, 1, 1]

        self.blocks = []
        last_dim = 1
        for i in range(5):
            self.blocks.append(
                SimpleAudioCNNBlock(
                    last_dim,
                    nb_filters[i],
                    kernel_sizes[i],
                    strides[i],
                    paddings[i],
                )
            )
            last_dim = nb_filters[i]

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = super().forward(X)

        Z = Z.unsqueeze(1)
        # Z: (N, 1, L)

        Z = self.blocks(Z)

        return Z
