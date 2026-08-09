from dataclasses import dataclass, field
from typing import List
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual module.

    Attributes:
        expansion (int): Expansion factor of the number of output channels.

        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolution.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolution.
        conv3 (nn.Conv2d): Third convolutional layer.
        bn3 (nn.BatchNorm2d): Batch normalization layer after the third convolution.
        shortcut (Optional[nn.Sequential]): Residual connection module.
    """

    expansion = 4

    def __init__(self, in_size: int, out_size: int, stride: int = 1):
        """
        Initialize a BottleneckBlock module.

        Args:
            in_size (int): Number of input channels.
            out_size (int): Number of intermediate channels.
            stride (int): Stride for the convolution. Defaults to 1.

        Returns:
            None
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size)

        self.conv2 = nn.Conv2d(
            out_size,
            out_size,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_size)

        self.conv3 = nn.Conv2d(
            out_size,
            self.expansion * out_size,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * out_size)

        self.shortcut = None
        if stride != 1 or in_size != self.expansion * out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_size,
                    self.expansion * out_size,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_size),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, 4*C', H', W').
        """
        residual = X
        if self.shortcut:
            residual = self.shortcut(X)

        Z = F.relu(self.bn1(self.conv1(X)))
        Z = F.relu(self.bn2(self.conv2(Z)))
        Z = self.bn3(self.conv3(Z))

        Z += residual
        return F.relu(Z)


class TemporalStatsPooling(nn.Module):
    """
    Temporal Statistics Pooling (TSTP) module.

    Attributes:
        out_size (int): Pooling output size.
    """

    def __init__(self, in_size: int, **kwargs):
        """
        Initialize a TemporalStatsPooling module.

        Args:
            in_size (int): Number of input channels.

        Returns:
            None
        """
        super().__init__()

        self.out_size = 2 * in_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, 2*C*H).
        """
        mean = X.mean(dim=-1).flatten(start_dim=1)
        std = torch.sqrt(torch.var(X, dim=-1) + 1e-7).flatten(start_dim=1)
        return torch.cat((mean, std), dim=1)


class AttentiveStatsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASTP) module.

    Attributes:
        out_size (int): Pooling output size.
        global_context (bool): Whether to use global context.
        linear1 (nn.Conv1d): First convolution module.
        linear2 (nn.Conv1d): Second convolution module.
    """

    def __init__(
        self,
        in_size: int,
        attention_channels: int = 128,
        global_context: bool = True,
    ):
        """
        Initialize an AttentiveStatsPooling module.

        Args:
            in_size (int): Number of input channels.
            attention_channels (int): Number of attention channels. Defaults to 128.
            global_context (bool): Whether to use global context. Defaults to True.

        Returns:
            None
        """
        super().__init__()

        self.out_size = 2 * in_size
        self.global_context = global_context

        self.linear1 = nn.Conv1d(
            in_size * 3 if global_context else in_size,
            attention_channels,
            kernel_size=1,
        )
        self.linear2 = nn.Conv1d(attention_channels, in_size, kernel_size=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, 2*C*H).
        """
        if X.ndim == 4:
            X = X.reshape(X.size(0), X.size(1) * X.size(2), X.size(3))

        if self.global_context:
            mean = torch.mean(X, dim=-1, keepdim=True).expand_as(X)
            std = torch.sqrt(
                torch.var(X, dim=-1, keepdim=True) + 1e-7
            ).expand_as(X)
            attn = torch.cat((X, mean, std), dim=1)
        else:
            attn = X

        # ReLU is not used here as it may be hard to converge.
        attn = torch.tanh(self.linear1(attn))
        attn = torch.softmax(self.linear2(attn), dim=2)

        mean = torch.sum(attn * X, dim=2)
        var = torch.sum(attn * (X**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))

        return torch.cat((mean, std), dim=1)


class PoolingModeEnum(Enum):
    """
    Enumeration representing the different pooling modes for ResNet293 encoder.

    Attributes:
        TSTP: Temporal Statistics Pooling (TSTP).
        ASTP: Attentive Statistics Pooling (ASTP).
    """

    TSTP = "tstp"
    ASTP = "astp"


@dataclass
class ResNet293Config(BaseEncoderConfig):
    """
    ResNet293 encoder configuration.

    Attributes:
        encoder_dim (int): Output dimension of the encoder.
        mel_n_mels (int): Number of mel filterbanks for Mel Spectrogram.
        pooling (bool): Whether to apply temporal pooling.
        pooling_mode (PoolingModeEnum): Temporal pooling mode.
        attention_channels (int): Number of channels for Attentive Statistics Pooling (ASTP).
        global_context (bool): Whether to use global context for Attentive Statistics Pooling (ASTP).
        base_dim (int): Base dimension for the encoder.
        num_blocks (List[int]): Number of BottleneckBlock modules for each stage.
    """

    encoder_dim: int = 256

    mel_n_mels: int = 80

    pooling: bool = True
    pooling_mode: PoolingModeEnum = PoolingModeEnum.TSTP

    attention_channels: int = 128
    global_context: bool = True

    base_dim: int = 32

    num_blocks: List[int] = field(default_factory=lambda: [10, 20, 64, 3])


class ResNet293(BaseEncoder):
    """
    ResNet293 encoder.

    Paper:
        Wespeaker: A Research and Production Oriented Speaker Embedding Learning Toolkit
        *Hongji Wang, Chengdong Liang, Shuai Wang, Zhengyang Chen, Binbin Zhang,
        Xu Xiang, Yanlei Deng, Yanmin Qian*
        ICASSP 2023
        https://arxiv.org/abs/2210.17016

    Attributes:
        _POOLING_MODULES (Dict[PoolingModeEnum, nn.Module]): Dictionary mapping pooling modes
            to corresponding modules.

        pooling_enabled (bool): Whether to apply temporal pooling.
        conv (nn.Conv2d): First convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer after the first convolution.
        relu (nn.ReLU): Activation function after the first convolution.
        block1 (nn.Sequential): First stage of BottleneckBlock modules.
        block2 (nn.Sequential): Second stage of BottleneckBlock modules.
        block3 (nn.Sequential): Third stage of BottleneckBlock modules.
        block4 (nn.Sequential): Fourth stage of BottleneckBlock modules.
        pooling (nn.Module): Pooling module.
        fc (nn.Linear): Final fully-connected layer.
    """

    _POOLING_MODULES = {
        PoolingModeEnum.TSTP: TemporalStatsPooling,
        PoolingModeEnum.ASTP: AttentiveStatsPooling,
    }

    def __init__(self, config: ResNet293Config):
        """
        Initialize a ResNet293 encoder.

        Args:
            config (ResNet293Config): Encoder configuration.

        Returns:
            None
        """
        super().__init__(config)

        assert len(config.num_blocks) == 4
        assert config.mel_n_mels % 8 == 0

        self.pooling_enabled = config.pooling

        base_dim = config.base_dim

        self.in_size = base_dim

        self.conv = nn.Conv2d(
            1,
            base_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(base_dim)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = self.__make_block(config.num_blocks[0], base_dim, 1)
        self.block2 = self.__make_block(config.num_blocks[1], base_dim * 2, 2)
        self.block3 = self.__make_block(config.num_blocks[2], base_dim * 4, 2)
        self.block4 = self.__make_block(config.num_blocks[3], base_dim * 8, 2)

        # Frequency dimension is downsampled by 8 (three strided stages).
        stats_dim = (config.mel_n_mels // 8) * base_dim * 8 * BottleneckBlock.expansion

        self.pooling = None
        out_size = stats_dim
        if self.pooling_enabled:
            self.pooling = self._POOLING_MODULES[config.pooling_mode](
                stats_dim,
                attention_channels=config.attention_channels,
                global_context=config.global_context,
            )
            out_size = self.pooling.out_size

        self.fc = nn.Linear(out_size, self.encoder_dim)

    def __make_block(self, num_layers: int, out_size: int, stride: int) -> nn.Module:
        """
        Create a stage of BottleneckBlock modules.

        Args:
            num_layers (int): Number of BottleneckBlock modules.
            out_size (int): Number of intermediate channels.
            stride (int): Stride for the first BottleneckBlock module convolution.

        Returns:
            nn.Sequential: Stage of BottleneckBlock modules.
        """
        layers = []
        for stride in [stride] + [1] * (num_layers - 1):
            layers.append(BottleneckBlock(self.in_size, out_size, stride))
            self.in_size = out_size * BottleneckBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, L).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, D) or (N, D, L').
        """
        Z = super().forward(X)
        # Z: (N, D, L) = (N, 80, 200)

        Z = Z.unsqueeze(1)
        # Z: (N, C, H, W) = (N, 1, 80, 200)

        Z = self.relu(self.bn(self.conv(Z)))

        Z = self.block1(Z)
        Z = self.block2(Z)
        Z = self.block3(Z)
        Z = self.block4(Z)
        # Z: (N, C, H, W) = (N, 1024, 10, 25)

        if self.pooling_enabled:
            Z = self.pooling(Z)
            Z = self.fc(Z)
        else:
            N, C, H, W = Z.size()
            Z = Z.reshape((N, C * H, W))
            Z = self.fc(Z.transpose(1, 2)).transpose(1, 2)

        return Z
