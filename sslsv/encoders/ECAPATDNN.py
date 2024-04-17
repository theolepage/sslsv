from dataclasses import dataclass, field
from typing import List, Tuple

import math

import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


class Conv1dSamePaddingReflect(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine padding
        L_in = x.size(-1)
        L_out = (
            math.floor(
                (L_in - self.dilation * (self.kernel_size - 1) - 1) / self.stride
            )
            + 1
        )
        padding = (L_in - L_out) // 2

        x = F.pad(x, (padding, padding), mode="reflect")

        return self.conv(x)


class TDNNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()

        self.conv = Conv1dSamePaddingReflect(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):

    def __init__(self, in_channels: int, se_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = Conv1dSamePaddingReflect(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Conv1dSamePaddingReflect(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class SERes2NetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1dSamePaddingReflect(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)

        return x + residual


class AttentiveStatisticsPooling(nn.Module):

    def __init__(
        self,
        channels: int,
        attention_channels: int = 128,
        global_context: bool = True,
    ):
        super().__init__()

        self.global_context = global_context

        in_channels = channels * 3 if global_context else channels

        self.tdnn = TDNNBlock(in_channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1dSamePaddingReflect(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def _compute_statistics(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = (m * x).sum(dim=2)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim=2)).pow(2)).sum(dim=2).clamp(eps))
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_context:
            L = x.size(-1)
            mean, std = self._compute_statistics(x, 1 / L)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = F.softmax(attn, dim=2)

        mean, std = self._compute_statistics(x, attn)

        stats = torch.cat((mean, std), dim=1)
        stats = stats.unsqueeze(dim=2)

        return stats


@dataclass
class ECAPATDNNConfig(BaseEncoderConfig):

    pooling: bool = True

    channels: List[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])

    kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])

    dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])

    attention_channels: int = 128

    res2net_scale: int = 8

    se_channels: int = 128

    global_context: bool = True


class ECAPATDNN(BaseEncoder):

    def __init__(self, config: ECAPATDNNConfig):
        super().__init__(config)

        self.pooling = config.pooling

        self.blocks = nn.ModuleList()

        self.blocks.append(
            TDNNBlock(
                config.mel_n_mels,
                config.channels[0],
                config.kernel_sizes[0],
                config.dilations[0],
            )
        )

        for i in range(1, len(config.channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    config.channels[i - 1],
                    config.channels[i],
                    res2net_scale=config.res2net_scale,
                    se_channels=config.se_channels,
                    kernel_size=config.kernel_sizes[i],
                    dilation=config.dilations[i],
                )
            )

        self.mfa = TDNNBlock(
            config.channels[-1],
            config.channels[-1],
            config.kernel_sizes[-1],
            config.dilations[-1],
        )

        self.asp = AttentiveStatisticsPooling(
            config.channels[-1],
            attention_channels=config.attention_channels,
            global_context=config.global_context,
        )
        self.asp_bn = nn.BatchNorm1d(config.channels[-1] * 2)

        last_in_channels = (
            config.channels[-1] * 2 if self.pooling else config.channels[-1]
        )
        self.fc = Conv1dSamePaddingReflect(
            in_channels=last_in_channels, out_channels=config.encoder_dim, kernel_size=1
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = super().forward(X)
        # Z shape: (B, C, L) = (B, 40, 200)

        feats = []
        for layer in self.blocks:
            Z = layer(Z)
            feats.append(Z)

        # Multi-layer feature aggregation
        Z = torch.cat(feats[1:], dim=1)
        Z = self.mfa(Z)

        # Attentive Statistical Pooling
        if self.pooling:
            Z = self.asp(Z)
            Z = self.asp_bn(Z)

        # Final linear transformation
        Z = self.fc(Z)

        if self.pooling:
            Z = Z.squeeze(dim=2)

        return Z
