from dataclasses import dataclass, field
from typing import List, Tuple

import math

import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


class ConvBlock(nn.Module):
    """
    ConvBlock module: Conv1d -> ReLU -> BN.

    Attributes:
        conv (nn.Conv1d): Convolution module.
        activation (nn.ReLU): Activation module.
        norm (nn.BatchNorm1d): Normalization module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        """
        Initialize a ConvBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel. Defaults to 1.
            stride (int): Stride of the convolution. Defaults to 1.
            padding (int): Padding of the convolution. Defaults to 0.
            dilation (int): Dilation rate for the convolution. Defaults to 1.

        Returns:
            None
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(nn.Module):
    """
    Res2Net module.

    Attributes:
        scale (int): Scale factor for the number of channels.
        width (int): Number of channels.
        nums (int): Number of Conv blocks.
        blocks (nn.ModuleList): List of ConvBlock modules.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        scale: int = 4,
    ):
        """
        Initialize a Res2NetBlock module.

        Args:
            channels (int): Number of input and output channels.
            kernel_size (int): Size of the kernel for the Conv blocks. Defaults to 1.
            stride (int): Stride for the Conv blocks. Defaults to 1.
            padding (int): Padding for the Conv blocks. Defaults to 0.
            dilation (int): Dilation factor for the Conv blocks. Defaults to 1.
            scale (int): Scale factor for the number of channels. Defaults to 4.

        Raises:
            AssertionError: If number of channels is not divisible by the scale factor.
        """
        super().__init__()

        assert channels % scale == 0

        self.scale = scale

        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.blocks = nn.ModuleList(
            [
                ConvBlock(
                    self.width,
                    self.width,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
                for i in range(self.nums)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = []

        spx = torch.split(x, self.width, dim=1)
        sp = spx[0]

        for i, block in enumerate(self.blocks):
            if i >= 1:
                sp = sp + spx[i]
            sp = block(sp)
            out.append(sp)

        if self.scale !=  1:
            out.append(spx[self.nums])

        out = torch.cat(out, dim=1)
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) module.

    Attributes:
        linear1 (nn.Linear): First linear module.
        linear2 (nn.Linear): Second linear module.
        relu (nn.ReLU): First activation module.
        sigmoid (nn.Sigmoid): Second activation module.
    """

    def __init__(self, channels: int, se_bottleneck_dim: int = 128):
        """
        Initialize a SEBlock module.

        Args:
            channels (int): Number of input and output channels.
            se_bottleneck_dim (int): SE bottleneck dimension. Defaults to 128.

        Returns:
            None
        """
        super().__init__()

        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        s = x.mean(dim=2)
        s = self.relu(self.linear1(s))
        s = self.sigmoid(self.linear2(s))
        return x * s.unsqueeze(2)


class SERes2NetBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Res2Net module.

    Attributes:
        block (nn.Sequential): SERes2Net module.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        scale: int,
    ):
        """
        Initialize a SERes2NetBlock module.

        Args:
            channels (int): Number of input and output channels.
            kernel_size (int): Size of the kernel for Res2Net convolution.
            stride (int): Stride for the Res2Net module convolution.
            padding (int): Padding for the Res2Net module convolution.
            dilation (int): Dilation rate for the Res2Net module convolution.
            scale (int): Scale factor for the number of channels in Res2Net.

        Returns:
            None
        """
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Res2NetBlock(
                channels,
                kernel_size,
                stride,
                padding,
                dilation,
                scale=scale
            ),
            ConvBlock(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            SEBlock(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x + self.block(x)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP) module.

    Attributes:
        global_context (bool): Whether to use global context.
        linear1 (nn.Conv1d): First linear module.
        linear2 (nn.Conv1d): Second linear module.
    """

    def __init__(
        self,
        channels: int,
        global_context: bool = True,
        attention_channels: int = 128,
    ):
        """
        Initialize an AttentiveStatisticsPooling module.

        Args:
            channels (int): Number of input channels.
            global_context (bool): Whether to use global context. Defaults to True.
            attention_channels (int): Number of attention channels. Defaults to 128.

        Returns:
            None
        """
        super().__init__()

        self.global_context = global_context

        in_channels = channels * 3 if global_context else channels

        self.linear1 = nn.Conv1d(in_channels, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor. Shape: (N, D, L).

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.global_context:
            mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            attn = torch.cat((x, mean, std), dim=1)
        else:
            attn = x

        alpha = torch.tanh(self.linear1(attn))
        alpha = F.softmax(self.linear2(alpha), dim=2)

        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-7))

        return torch.cat((mean, std), dim=1)


@dataclass
class ECAPATDNNConfig(BaseEncoderConfig):
    """
    ECAPA-TDNN encoder configuration.

    Attributes:
        pooling (bool): Whether to apply temporal pooling.
        channels (int): Number of channels.
        global_context (bool): Whether to use global context for Attentive Statistics Pooling (ASP).
    """

    pooling: bool = True

    channels: int = 512

    global_context: bool = True


class ECAPATDNN(BaseEncoder):
    """
    Emphasized Channel Attention, Propagation and Aggregation in TDNN (ECAPA-TDNN) encoder.

    Paper:
        ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
        *Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck*
        INTERSPEECH 2020
        https://arxiv.org/abs/2005.07143

    Attributes:
        pooling (bool): Whether to apply temporal pooling.
        layer1 (ConvBlock): First layer.
        layer2 (SERes2NetBlock): Second layer.
        layer3 (SERes2NetBlock): Third layer.
        layer4 (SERes2NetBlock): Fourth layer.
        mfa_conv (nn.Conv1d): Multi-Frame Aggregation (MFA) module.
        mfa_relu (nn.ReLU): MFA activation.
        asp (AttentiveStatisticsPooling): Attentive Statistics Pooling (ASP) module.
        asp_bn (nn.BatchNorm1d): Batch normalization module for Attentive Statistics Pooling (ASP).
        fc (nn.Linear): Final fully-connected layer.
        last_bn (nn.BatchNorm1d): Final Batch normalization layer.
    """

    def __init__(self, config: ECAPATDNNConfig):
        """
        Initialize an ECAPA-TDNN encoder.

        Args:
            config (ECAPATDNNConfig): Encoder configuration.

        Returns:
            None
        """
        super().__init__(config)

        self.pooling = config.pooling

        self.layer1 = ConvBlock(
            config.mel_n_mels,
            config.channels,
            kernel_size=5,
            padding=2
        )

        self.layer2 = SERes2NetBlock(
            config.channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8
        )

        self.layer3 = SERes2NetBlock(
            config.channels,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8
        )

        self.layer4 = SERes2NetBlock(
            config.channels,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8
        )

        cat_channels = config.channels * 3
        out_channels = 512 * 3

        # MFA
        self.mfa_conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.mfa_relu = nn.ReLU()

        # Pooling
        pooling_out_dim = out_channels * 2
        self.asp = AttentiveStatisticsPooling(out_channels, config.global_context)
        self.asp_bn = nn.BatchNorm1d(pooling_out_dim)

        last_in_channels = (
            pooling_out_dim if self.pooling else out_channels
        )
        self.fc = nn.Linear(last_in_channels, config.encoder_dim)

        self.last_bn = nn.BatchNorm1d(config.encoder_dim)

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

        out1 = self.layer1(Z)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # Multi-layer Feature Aggregation
        Z = torch.cat((out2, out3, out4), dim=1)
        Z = self.mfa_relu(self.mfa_conv(Z))

        # Attentive Statistical Pooling
        if self.pooling:
            Z = self.asp(Z)
            Z = self.asp_bn(Z)

        # Final linear transformation
        Z = self.fc(Z)

        Z = self.last_bn(Z)

        return Z
