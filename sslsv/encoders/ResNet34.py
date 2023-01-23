import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


class ResNetBlock(nn.Module):

    def __init__(self, in_size, out_size, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_size,
            out_size,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_size)

        self.conv2 = nn.Conv2d(
            out_size,
            out_size,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_size)

        self.se = SELayer(out_size)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, X):
        residual = X
        if self.downsample: residual = self.downsample(residual)

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

    def __init__(self, in_size, reduction=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_size, in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_size // reduction, in_size),
            nn.Sigmoid()
        )

    def forward(self, X):
        b, c, _, _ = X.size()

        Y = self.fc(self.avg_pool(X).view(b, c)).view(b, c, 1, 1)
        return X * Y


class SelfAttentivePooling(nn.Module):

    def __init__(self, out_size, dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(out_size, dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, out_size, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, X):
        b, c, h, w = X.size()

        X = X.reshape(b, -1, w)
        W = self.attention(X)
        return torch.sum(W * X, dim=2)


class StatsPooling(nn.Module):

    def __init__(self, out_size):
        super().__init__()

    def forward(self, X):
        mean = torch.mean(X, dim=-1)
        std = torch.std(X, dim=-1)
        stats = torch.cat((mean, std), dim=1)
        return stats


@dataclass
class ResNet34Config(BaseEncoderConfig):

    pooling: bool = True
    pooling_mode: str = 'sap'

    base_dim: int = 32


class ResNet34(BaseEncoder):

    _POOLING_MODULES = {
        'sap': SelfAttentivePooling,
        'stats': StatsPooling
    }

    def __init__(self, config):
        super().__init__(config)

        self.pooling = config.pooling

        base_dim = config.base_dim

        self.conv = nn.Conv2d(1, base_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(base_dim)

        self.block1 = self.__make_block(3, base_dim, base_dim, 1)
        self.block2 = self.__make_block(4, base_dim, base_dim * 2, 2)
        self.block3 = self.__make_block(6, base_dim * 2, base_dim * 4, 2)
        self.block4 = self.__make_block(3, base_dim * 4, base_dim * 8, 2)

        out_size = int(config.mel_n_mels / 8 * (base_dim * 8))
        
        self.pooling = self._POOLING_MODULES[config.pooling_mode](out_size)

        if config.pooling_mode == 'stats': out_size *= 2

        self.fc = nn.Linear(out_size, self.encoder_dim)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_block(self, num_layers, in_size, out_size, stride):
        layers = []
        layers.append(ResNetBlock(in_size, out_size, stride))
        for i in range(1, num_layers):
            layers.append(ResNetBlock(out_size, out_size, 1))
        return nn.Sequential(*layers)

    def forward(self, X):
        Z = super().forward(X)
        # Z: (B, C, L) = (B, 40, 200)

        Z = Z.unsqueeze(1)
        # Z: (B, C, H, W) = (B, 1, 40, 200)

        Z = self.conv(Z)
        Z = self.relu(Z)
        Z = self.bn(Z)

        Z = self.block1(Z)
        Z = self.block2(Z)
        Z = self.block3(Z)
        Z = self.block4(Z)

        if self.pooling:
            Z = self.pooling(Z)
            Z = self.fc(Z)
        else:
            B, C, H, W = Z.size()
            Z = Z.reshape((B, -1, W))
            Z = self.fc(Z.transpose(1, 2)).transpose(2, 1)

        return Z