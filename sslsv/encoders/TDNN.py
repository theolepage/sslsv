import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import List

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


@dataclass
class TDNNConfig(BaseEncoderConfig):

    nb_blocks: int = 5

    channels: List[int] = field(
        default_factory=lambda: [512, 512, 512, 512, 1500]
    )
    
    kernel_sizes: List[int] = field(
        default_factory=lambda: [5, 3, 3, 1, 1]
    )
    
    dilations: List[int] = field(
        default_factory=lambda: [1, 2, 3, 1, 1]
    )


class TDNNBlock(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, dilation):
        super().__init__()

        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, X):
        return self.bn(self.activation(self.conv(X)))


class TDNN(BaseEncoder):

    def __init__(self, config):
        super().__init__(config)

        self.blocks = nn.Sequential(*[
            TDNNBlock(
                in_dim=config.channels[i - 1] if i > 0 else config.mel_n_mels,
                out_dim=config.channels[i],
                kernel_size=config.kernel_sizes[i],
                dilation=config.dilations[i]
            )
            for i in range(config.nb_blocks)
        ])

        self.last_fc = nn.Linear(
            config.channels[-1] * 2,
            config.encoder_dim
        )

    def forward(self, X):
        Z = super().forward(X)
        # Z shape: (B, C, L) = (B, 40, 200)

        Z = self.blocks(Z)

        # Stats pooling
        mean = torch.mean(Z, dim=-1)
        std = torch.std(Z, dim=-1)
        stats = torch.cat((mean, std), dim=1)

        Z = self.last_fc(stats)

        return Z