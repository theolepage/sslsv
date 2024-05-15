from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


class ResNetBlock(nn.Module):
    """
    ResNetBlock module.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolution.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolution.
        se (SELayer): Squeeze-and-Excitation layer.
        relu (nn.ReLU): Activation function.
        downsample (nn.Sequential): Downsample module.
    """

    def __init__(self, in_size: int, out_size: int, stride: int):
        """
        Initialize a ResNetBlock module.

        Args:
            in_size (int): Number of input channels.
            out_size (int): Number of output channels.
            stride (int): Stride for the convolution.

        Returns:
            None
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_size,
            out_size,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_size)

        self.conv2 = nn.Conv2d(
            out_size,
            out_size,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_size)

        self.se = SELayer(out_size)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_size != out_size:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = X
        if self.downsample:
            residual = self.downsample(residual)

        Z = self.conv1(X)
        Z = self.relu(Z)
        Z = self.bn1(Z)

        Z = self.conv2(Z)
        Z = self.bn2(Z)
        Z = self.se(Z)

        Z += residual
        Z = self.relu(Z)
        return Z


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) module.

    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling module.
        fc (nn.Sequential): Final fully-connected layer.
    """

    def __init__(self, in_size: int, reduction: int = 8):
        """
        Initialize an SELayer module.

        Args:
            in_size (int): Number of input channels.
            reduction (int): Reduction factor for the bottleneck architecture. Defaults to 8.

        Returns:
            None
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_size, in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_size // reduction, in_size),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        b, c, _, _ = X.size()

        Y = self.fc(self.avg_pool(X).view(b, c)).view(b, c, 1, 1)
        return X * Y


class SelfAttentivePooling(nn.Module):
    """
    Self-Attentive Pooling (SAP) module.

    Attributes:
        attention (nn.Parameter): Attention parameter.
        sap_linear (int): SAP linear module.
    """

    def __init__(self, out_size: int):
        """
        Initialize a Self Attentive Pooling (SAP) module.

        Args:
            out_size (int): Encoder output length.

        Returns:
            None
        """
        super().__init__()

        self.attention = nn.Parameter(torch.FloatTensor(out_size, 1))
        nn.init.xavier_normal_(self.attention)

        self.sap_linear = nn.Linear(out_size, out_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, D, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, D).
        """
        X = X.mean(dim=2).transpose(1, 2)  # Shape: (N, L, D)
        W = torch.tanh(self.sap_linear(X)) @ self.attention  # Shape: (N, L, 1)
        W = F.softmax(W, dim=1)
        return torch.sum(X * W, dim=1)


class StatsPooling(nn.Module):
    """
    Statistics Pooling.

    Attributes:
        out_size (int): Encoder output size.
    """

    def __init__(self, out_size: int):
        """
        Initialize a Statistics Pooling module.

        Args:
            out_size (int): Encoder output size.

        Returns:
            None
        """
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, D, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, 2*D).
        """
        X = X.view(X.size(0), -1, X.size(-1))  # Shape: (N, D, L)

        mean = torch.mean(X, dim=-1, keepdim=True)
        std = torch.sqrt(
            torch.mean((X - mean) ** 2, dim=-1, keepdim=True).clamp(min=1e-5)
        )
        stats = torch.cat((mean, std), dim=1).squeeze(dim=-1)

        return stats


class PoolingModeEnum(Enum):
    """
    Enumeration representing the different pooling modes for ResNet34 encoder.

    Attributes:
        NONE: No pooling mode.
        SAP: Self-Attentive Pooling (SAP).
        STATS: Statistics Pooling.
    """

    NONE = None
    SAP = "sap"
    STATS = "stats"


@dataclass
class ResNet34Config(BaseEncoderConfig):
    """
    ResNet34 encoder configuration.

    Attributes:
        pooling (bool): Whether to apply temporal pooling.
        pooling_mode (PoolingModeEnum): Temporal pooling mode.
        base_dim (int): Base dimension for the encoder.
        enable_last_bn (bool): True if the last batch normalization is enabled, False otherwise.
    """

    pooling: bool = True
    pooling_mode: PoolingModeEnum = PoolingModeEnum.SAP

    base_dim: int = 16

    enable_last_bn: bool = False


class ResNet34(BaseEncoder):
    """
    ResNet34 encoder.

    Paper:
        VoxCeleb2: Deep Speaker Recognition
        *Joon Son Chung, Arsha Nagrani, Andrew Zisserman*
        INTERSPEECH 2018
        https://arxiv.org/abs/1806.05622

    Attributes:
        _POOLING_MODULES (Dict[PoolingModeEnum, nn.Module]): Dictionary mapping pooling modes
            to corresponding modules.

        conv (nn.Conv2d): First convolutional layer.
        relu (nn.ReLU): Activation function after the first convolution.
        bn (nn.BatchNorm2d): Batch normalization layer after the first convolution.
        block1 (nn.Sequential): First block of ResNetBlock modules.
        block2 (nn.Sequential): Second block of ResNetBlock modules.
        block3 (nn.Sequential): Third block of ResNetBlock modules.
        block4 (nn.Sequential): Fourth block of ResNetBlock modules.
        out_size (int): Encoder output length.
        pooling (nn.Module): Pooling module.
        fc (nn.Linear): Final fully-connected layer.
        last_bn (nn.BatchNorm1d): Batch normalization for the last layer.
    """

    _POOLING_MODULES = {
        PoolingModeEnum.SAP: SelfAttentivePooling,
        PoolingModeEnum.STATS: StatsPooling,
    }

    def __init__(self, config: ResNet34Config):
        """
        Initialize a ResNet34 encoder.

        Args:
            config (ResNet34Config): Encoder configuration.

        Returns:
            None
        """
        super().__init__(config)

        base_dim = config.base_dim

        self.conv = nn.Conv2d(
            1,
            base_dim,
            kernel_size=7,
            stride=(2, 1),
            padding=3,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(base_dim)

        self.block1 = self.__make_block(3, base_dim, base_dim, 1)
        self.block2 = self.__make_block(4, base_dim, base_dim * 2, 2)
        self.block3 = self.__make_block(6, base_dim * 2, base_dim * 4, 2)
        self.block4 = self.__make_block(3, base_dim * 4, base_dim * 8, 1)

        out_size = base_dim * 8

        if config.pooling_mode == PoolingModeEnum.STATS:
            out_size = 2 * int(config.mel_n_mels / 8 * (base_dim * 8))
        if not config.pooling:
            out_size = int(config.mel_n_mels / 8 * (base_dim * 8))

        self.pooling = None
        if config.pooling:
            self.pooling = self._POOLING_MODULES[config.pooling_mode](out_size)

        self.fc = nn.Linear(out_size, self.encoder_dim)

        self.last_bn = (
            nn.BatchNorm1d(self.encoder_dim) if config.enable_last_bn else None
        )

        self.__init_weights()

    def __init_weights(self):
        """
        Initialize the weights of convolutional and batch normalization layers.

        Returns:
            None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_block(
        self,
        num_layers: int,
        in_size: int,
        out_size: int,
        stride: int,
    ) -> nn.Module:
        """
        Create a block of ResNetBlock modules.

        Args:
            num_layers (int): Number of ResNetBlock modules.
            in_size (int): Number of input channels.
            out_size (int): Number of output channels.
            stride (int): Stride for the first ResNetBlock module convolution.

        Returns:
            nn.Sequential: Block of ResNetBlock modules.
        """
        layers = []
        layers.append(ResNetBlock(in_size, out_size, stride))
        for i in range(1, num_layers):
            layers.append(ResNetBlock(out_size, out_size, 1))
        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, L).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, D).
        """
        Z = super().forward(X)
        # Z: (N, D, L) = (N, 40, 200)

        Z = Z.unsqueeze(1)
        # Z: (N, D, H, W) = (N, 1, 40, 200)

        Z = self.conv(Z)
        Z = self.bn(Z)
        Z = self.relu(Z)

        Z = self.block1(Z)
        Z = self.block2(Z)
        Z = self.block3(Z)
        Z = self.block4(Z)

        if self.pooling:
            Z = self.pooling(Z)
            Z = self.fc(Z)
        else:
            N, D, H, W = Z.size()
            Z = Z.reshape((N, -1, W))
            Z = self.fc(Z.transpose(1, 2)).transpose(2, 1)

        if self.last_bn:
            Z = self.last_bn(Z)

        return Z
