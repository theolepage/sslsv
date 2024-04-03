import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
from enum import Enum

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
        if stride != 1 or in_size != out_size:
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

        #self.attention = nn.Sequential(
        #    nn.Conv1d(out_size, dim, kernel_size=1),
        #    nn.ReLU(),
        #    nn.BatchNorm1d(dim),
        #    nn.Conv1d(dim, out_size, kernel_size=1),
        #    nn.Softmax(dim=2)
        #)

        self.attention = nn.Parameter(torch.FloatTensor(out_size, 1))
        nn.init.xavier_normal_(self.attention)

        self.sap_linear = nn.Linear(out_size, out_size)

    def forward(self, X):
        b, c, h, w = X.size()
        # B 1 40 200

        X = X.mean(dim=2).transpose(1, 2) # B W C
        W = torch.tanh(self.sap_linear(X)) @ self.attention # B W 1
        W = F.softmax(W, dim=1)
        return torch.sum(X * W, dim=1)

        #X = X.reshape(b, -1, w)
        #W = self.attention(X)
        #return torch.sum(W * X, dim=2)


class StatsPooling(nn.Module):

    def __init__(self, out_size):
        super().__init__()

    def forward(self, X):
        X = X.view(X.size(0), -1, X.size(-1)) # B C L

        mean = torch.mean(X, dim=-1, keepdim=True)
        std = torch.sqrt(torch.mean((X - mean) ** 2, dim=-1, keepdim=True).clamp(min=1e-5))
        stats = torch.cat((mean, std), dim=1).squeeze(dim=-1)

        return stats


class PoolingModeEnum(Enum):

    NONE  = None
    SAP   = 'sap'
    STATS = 'stats'


@dataclass
class ResNet34Config(BaseEncoderConfig):

    pooling: bool = True
    pooling_mode: PoolingModeEnum = PoolingModeEnum.SAP

    base_dim: int = 16

    enable_last_bn: bool = False


class ResNet34(BaseEncoder):

    _POOLING_MODULES = {
        PoolingModeEnum.SAP: SelfAttentivePooling,
        PoolingModeEnum.STATS: StatsPooling
    }

    def __init__(self, config):
        super().__init__(config)

        self.pooling = config.pooling

        base_dim = config.base_dim

        self.conv = nn.Conv2d(1, base_dim, kernel_size=7, stride=(2, 1), padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(base_dim)

        self.block1 = self.__make_block(3, base_dim, base_dim, 1)
        self.block2 = self.__make_block(4, base_dim, base_dim * 2, 2)
        self.block3 = self.__make_block(6, base_dim * 2, base_dim * 4, 2)
        self.block4 = self.__make_block(3, base_dim * 4, base_dim * 8, 1)

        out_size = base_dim * 8
        
        if config.pooling_mode == PoolingModeEnum.STATS:
            out_size = 2 * int(80 / 8 * (base_dim * 8))
        if not config.pooling:
            out_size = int(40 / 8 * (base_dim * 8))

        self.pooling = None
        if config.pooling:
            self.pooling = self._POOLING_MODULES[config.pooling_mode](out_size)

        self.fc = nn.Linear(out_size, self.encoder_dim)

        self.last_bn = (
            nn.BatchNorm1d(self.encoder_dim)
            if config.enable_last_bn else None
        )

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
            B, C, H, W = Z.size()
            Z = Z.reshape((B, -1, W))
            Z = self.fc(Z.transpose(1, 2)).transpose(2, 1)
        
        if self.last_bn: Z = self.last_bn(Z)

        return Z