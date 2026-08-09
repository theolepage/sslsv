from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

import math

import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig

# ReDimNet2 relies on the same Attentive Statistics Pooling (ASTP) module as the
# wespeaker-based encoders.
from sslsv.encoders.ResNet293 import AttentiveStatsPooling


class LayerNorm(nn.Module):
    """
    Layer normalization module operating on the channel dimension of a
    channels-first tensor.

    Attributes:
        weight (nn.Parameter): Scale parameter.
        bias (nn.Parameter): Shift parameter.
        eps (float): Small value to prevent division by zero.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        """
        Initialize a LayerNorm module.

        Args:
            channels (int): Number of channels.
            eps (float): Small value to prevent division by zero. Defaults to 1e-6.

        Returns:
            None
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, ...).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, ...).
        """
        mean = X.mean(dim=1, keepdim=True)
        var = (X - mean).pow(2).mean(dim=1, keepdim=True)
        Z = (X - mean) / torch.sqrt(var + self.eps)

        weight, bias = self.weight, self.bias
        for _ in range(X.ndim - 2):
            weight = weight.unsqueeze(-1)
            bias = bias.unsqueeze(-1)

        return weight * Z + bias


class fwSEBlock(nn.Module):
    """
    Frequency-wise Squeeze-and-Excitation (fwSE) module.

    Attributes:
        squeeze (nn.Linear): Squeeze module.
        excitation (nn.Linear): Excitation module.
        relu (nn.ReLU): Activation module.
    """

    def __init__(self, nb_freqs: int, se_channels: int = 64):
        """
        Initialize a fwSEBlock module.

        Args:
            nb_freqs (int): Number of frequency bins.
            se_channels (int): Number of SE channels. Defaults to 64.

        Returns:
            None
        """
        super().__init__()

        self.squeeze = nn.Linear(nb_freqs, se_channels)
        self.excitation = nn.Linear(se_channels, nb_freqs)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, F, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, F, T).
        """
        Z = torch.mean(X, dim=[1, 3])
        Z = self.relu(self.squeeze(Z))
        Z = torch.sigmoid(self.excitation(Z))
        return X * Z[:, None, :, None]


class ResBasicBlock(nn.Module):
    """
    Residual module with depthwise-separable convolutions.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv1pw (nn.Module): Pointwise layer after the first convolution.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolution.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv2pw (nn.Module): Pointwise layer after the second convolution.
        bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolution.
        relu (nn.ReLU): Activation module.
        se (nn.Module): Squeeze-and-Excitation module.
        downsample (nn.Module): Residual connection module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nb_freqs: int,
        stride: int = 1,
        se_channels: int = 64,
        group_divisor: Optional[int] = 1,
        use_fwSE: bool = False,
    ):
        """
        Initialize a ResBasicBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            nb_freqs (int): Number of frequency bins.
            stride (int): Stride of the first convolution. Defaults to 1.
            se_channels (int): Number of SE channels. Defaults to 64.
            group_divisor (Optional[int]): Divisor determining the number of groups of the
                convolutions. None to use dense convolutions. Defaults to 1.
            use_fwSE (bool): Whether to use a fwSE module. Defaults to False.

        Returns:
            None
        """
        super().__init__()

        grouped = group_divisor is not None

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels if grouped else out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=in_channels // group_divisor if grouped else 1,
        )
        self.conv1pw = (
            nn.Conv2d(in_channels, out_channels, 1) if grouped else nn.Identity()
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=out_channels // group_divisor if grouped else 1,
        )
        self.conv2pw = (
            nn.Conv2d(out_channels, out_channels, 1) if grouped else nn.Identity()
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.se = fwSEBlock(nb_freqs, se_channels) if use_fwSE else nn.Identity()

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, F, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C', F, T).
        """
        Z = self.conv1pw(self.conv1(X))
        Z = self.bn1(self.relu(Z))

        Z = self.conv2pw(self.conv2(Z))
        Z = self.se(self.bn2(Z))

        return self.relu(Z + self.downsample(X))


class ConvNeXtLikeBlock(nn.Module):
    """
    ConvNeXt-like module with multiple parallel depthwise convolutions.

    Attributes:
        dwconvs (nn.ModuleList): List of depthwise convolution modules.
        norm (nn.BatchNorm1d): Normalization module.
        act (nn.GELU): Activation module.
        pwconv (nn.Conv1d): Pointwise convolution module.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int],
        group_divisor: Optional[int] = 1,
    ):
        """
        Initialize a ConvNeXtLikeBlock module.

        Args:
            channels (int): Number of input and output channels.
            kernel_sizes (List[int]): Sizes of the kernels of the parallel convolutions.
            group_divisor (Optional[int]): Divisor determining the number of groups of the
                convolutions. None to use dense convolutions. Defaults to 1.

        Returns:
            None
        """
        super().__init__()

        self.dwconvs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding="same",
                    groups=(
                        channels // group_divisor if group_divisor is not None else 1
                    ),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.norm = nn.BatchNorm1d(channels * len(kernel_sizes))
        self.act = nn.GELU()
        self.pwconv = nn.Conv1d(channels * len(kernel_sizes), channels, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, T).
        """
        Z = torch.cat([dwconv(X) for dwconv in self.dwconvs], dim=1)
        Z = self.act(self.norm(Z))
        Z = self.pwconv(Z)
        return X + Z


class PosEncConv(nn.Module):
    """
    Convolutional positional encoding module.

    Attributes:
        conv (nn.Conv1d): Depthwise convolution module.
        norm (LayerNorm): Normalization module.
    """

    def __init__(self, channels: int, kernel_size: int = 59):
        """
        Initialize a PosEncConv module.

        Args:
            channels (int): Number of input and output channels.
            kernel_size (int): Size of the convolutional kernel. Defaults to 59.

        Returns:
            None
        """
        super().__init__()

        assert kernel_size % 2 == 1

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = LayerNorm(channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, T).
        """
        return X + self.norm(self.conv(X))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer operating on channels-first sequences.

    Attributes:
        attention (nn.MultiheadAttention): Multi-head self-attention module.
        norm (nn.LayerNorm): Normalization module after self-attention.
        feed_forward (nn.Sequential): Feed-forward module.
        final_norm (nn.LayerNorm): Normalization module after the feed-forward module.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        nb_heads: int = 4,
        eps: float = 1e-6,
    ):
        """
        Initialize a TransformerEncoderLayer module.

        Args:
            channels (int): Number of input and output channels.
            hidden_channels (int): Number of channels of the feed-forward module.
            nb_heads (int): Number of attention heads. Defaults to 4.
            eps (float): Small value to prevent division by zero. Defaults to 1e-6.

        Returns:
            None
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(channels, nb_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels, eps=eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_channels, channels),
        )
        self.final_norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, T).
        """
        Z = X.transpose(1, 2)

        Z = Z + self.attention(Z, Z, Z, need_weights=False)[0]
        Z = self.norm(Z)
        Z = Z + self.feed_forward(Z)
        Z = self.final_norm(Z)

        return Z.transpose(1, 2)


class Block1dTypeEnum(Enum):
    """
    Enumeration representing the different 1D blocks for ReDimNet2 encoder.

    Attributes:
        CONV: Sequence of ConvNeXt-like modules.
        ATT: Positional encoding and Transformer encoder layer.
        CONV_ATT: Sequence of ConvNeXt-like modules and a Transformer encoder layer.
    """

    CONV = "conv"
    ATT = "att"
    CONV_ATT = "conv+att"


class Block2dTypeEnum(Enum):
    """
    Enumeration representing the different 2D blocks for ReDimNet2 encoder.

    Attributes:
        BASIC_RESNET: Residual module.
        BASIC_RESNET_FWSE: Residual module with a fwSE module.
    """

    BASIC_RESNET = "basic_resnet"
    BASIC_RESNET_FWSE = "basic_resnet_fwse"


class TimeContextBlock1d(nn.Module):
    """
    Time context module operating on the 1D representation of the feature maps.

    Attributes:
        red_dim_conv (nn.Sequential): Dimensionality reduction module.
        tcm (nn.Sequential): Time context modeling module.
        exp_dim_conv (nn.Conv1d): Dimensionality expansion module.
    """

    _CONV_KERNEL_SIZES = [7, 19, 31, 59]

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        block_type: Block1dTypeEnum = Block1dTypeEnum.CONV_ATT,
        pos_kernel_size: int = 59,
    ):
        """
        Initialize a TimeContextBlock1d module.

        Args:
            channels (int): Number of input and output channels.
            hidden_channels (int): Number of channels of the time context modeling module.
            block_type (Block1dTypeEnum): Type of the time context modeling module.
                Defaults to Block1dTypeEnum.CONV_ATT.
            pos_kernel_size (int): Size of the positional encoding kernel. Defaults to 59.

        Returns:
            None
        """
        super().__init__()

        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1),
            LayerNorm(hidden_channels),
        )

        convs = [
            ConvNeXtLikeBlock(hidden_channels, kernel_sizes=[kernel_size])
            for kernel_size in self._CONV_KERNEL_SIZES
        ]

        if block_type == Block1dTypeEnum.CONV:
            self.tcm = nn.Sequential(*convs)
        elif block_type == Block1dTypeEnum.ATT:
            self.tcm = nn.Sequential(
                PosEncConv(hidden_channels, kernel_size=pos_kernel_size),
                TransformerEncoderLayer(hidden_channels, hidden_channels * 2),
            )
        elif block_type == Block1dTypeEnum.CONV_ATT:
            self.tcm = nn.Sequential(
                *convs,
                TransformerEncoderLayer(hidden_channels, hidden_channels),
            )
        else:
            raise Exception(f"1D block type {block_type} is not handled")

        self.exp_dim_conv = nn.Conv1d(hidden_channels, channels, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, T).
        """
        return X + self.exp_dim_conv(self.tcm(self.red_dim_conv(X)))


class To1d(nn.Module):
    """
    Module reshaping a 2D feature map into its 1D representation.
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, F, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C*F, T).
        """
        N, C, F_, T = X.size()
        return X.permute((0, 2, 1, 3)).reshape((N, C * F_, T))


class TimeUpsample(nn.Module):
    """
    Module upsampling the time dimension by an integer factor (nearest neighbor).

    Implemented with an expand/reshape instead of `nn.Upsample` as
    `upsample_nearest1d` has no bfloat16 kernel before PyTorch 2.x.

    Attributes:
        scale_factor (int): Upsampling factor.
    """

    def __init__(self, scale_factor: int):
        """
        Initialize a TimeUpsample module.

        Args:
            scale_factor (int): Upsampling factor.

        Returns:
            None
        """
        super().__init__()

        self.scale_factor = scale_factor

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, T*scale_factor).
        """
        if self.scale_factor == 1:
            return X

        N, C, T = X.size()
        Z = X.unsqueeze(-1).expand(N, C, T, self.scale_factor)
        return Z.reshape(N, C, T * self.scale_factor)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.

        Returns:
            str: Extra representation.
        """
        return f"scale_factor={self.scale_factor}"


class To2d(nn.Module):
    """
    Module reshaping a 1D representation back into a 2D feature map.

    Attributes:
        nb_freqs (int): Number of frequency bins.
        channels (int): Number of channels.
    """

    def __init__(self, nb_freqs: int, channels: int):
        """
        Initialize a To2d module.

        Args:
            nb_freqs (int): Number of frequency bins.
            channels (int): Number of channels.

        Returns:
            None
        """
        super().__init__()

        self.nb_freqs = nb_freqs
        self.channels = channels

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, C*F, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, F, T).
        """
        N, _, T = X.size()
        Z = X.reshape((N, self.nb_freqs, self.channels, T))
        return Z.permute((0, 2, 1, 3))


class FeatureMapWeighting(nn.Module):
    """
    Module computing a learnable weighted sum of 1D feature maps.

    Attributes:
        w (nn.Parameter): Feature maps weights.
    """

    def __init__(self, nb_feature_maps: int, channels: int):
        """
        Initialize a FeatureMapWeighting module.

        Args:
            nb_feature_maps (int): Number of feature maps to aggregate.
            channels (int): Number of channels.

        Returns:
            None
        """
        super().__init__()

        self.w = nn.Parameter(
            torch.zeros(1, nb_feature_maps, channels, 1),
            requires_grad=nb_feature_maps > 1,
        )

    def forward(self, X: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (List[torch.Tensor]): Input tensors. Shape: (N, C, T).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, C, T).
        """
        w = F.softmax(self.w, dim=1)
        return (w * torch.stack(X, dim=1)).sum(dim=1)


@dataclass
class ReDimNet2StageConfig:
    """
    ReDimNet2 stage configuration.

    Attributes:
        freq_stride (int): Stride along the frequency dimension.
        time_stride (int): Stride along the time dimension.
        nb_blocks (int): Number of 2D blocks.
        channels_expansion (float): Expansion factor of the number of channels of the
            2D blocks.
        attention_reduction (Optional[int]): Reduction factor determining the number of
            channels of the 1D block. None to disable the 1D block.
    """

    freq_stride: int = 1
    time_stride: int = 1
    nb_blocks: int = 3
    channels_expansion: float = 1.0
    attention_reduction: Optional[int] = None


@dataclass
class ReDimNet2Config(BaseEncoderConfig):
    """
    ReDimNet2 encoder configuration.

    The default configuration corresponds to the ReDimNet2-B6 variant.

    Attributes:
        encoder_dim (int): Output dimension of the encoder.
        mel_n_mels (int): Number of mel filterbanks for Mel Spectrogram.
        pooling (bool): Whether to apply temporal pooling.
        base_channels (int): Base number of channels of the encoder.
        head_channels (Optional[int]): Number of channels of the final 2D projection.
            None to skip the final projection.
        stages (List[ReDimNet2StageConfig]): Configuration of each stage.
        block_1d_type (Block1dTypeEnum): Type of the 1D blocks.
        block_2d_type (Block2dTypeEnum): Type of the 2D blocks.
        group_divisor (Optional[int]): Divisor determining the number of groups of the
            2D convolutions. None to use dense convolutions.
        compress_tconvs (bool): Whether to use grouped strided convolutions.
        agg_group_norm (bool): Whether to normalize aggregated 1D feature maps.
        attention_channels (int): Number of channels for Attentive Statistics Pooling (ASTP).
        global_context (bool): Whether to use global context for Attentive Statistics
            Pooling (ASTP).
    """

    encoder_dim: int = 192

    mel_n_mels: int = 72

    pooling: bool = True

    base_channels: int = 64

    head_channels: Optional[int] = 224

    stages: List[ReDimNet2StageConfig] = field(
        default_factory=lambda: [
            ReDimNet2StageConfig(1, 1, 3, 3.00, 64),
            ReDimNet2StageConfig(2, 1, 4, 2.00, 64),
            ReDimNet2StageConfig(1, 2, 5, 2.00, 48),
            ReDimNet2StageConfig(2, 1, 5, 1.00, 48),
            ReDimNet2StageConfig(1, 2, 4, 0.75, 32),
            ReDimNet2StageConfig(2, 1, 3, 0.50, 24),
        ]
    )

    block_1d_type: Block1dTypeEnum = Block1dTypeEnum.CONV_ATT
    block_2d_type: Block2dTypeEnum = Block2dTypeEnum.BASIC_RESNET

    group_divisor: Optional[int] = 1

    compress_tconvs: bool = True

    agg_group_norm: bool = True

    attention_channels: int = 128

    global_context: bool = True


class ReDimNet2(BaseEncoder):
    """
    ReDimNet2 encoder.

    Paper:
        ReDimNet2: Scaling Speaker Verification via Time-Pooled Dimension Reshaping
        *Ivan Yakovlev, Anton Okhotnikov*
        INTERSPEECH 2026
        https://arxiv.org/abs/2603.11841

    Attributes:
        pooling (bool): Whether to apply temporal pooling.
        time_stride (int): Total stride along the time dimension.
        stem (nn.Sequential): Input module.
        stem_norm (nn.Module): Normalization module after the input module.
        stage_aggs (nn.ModuleList): List of feature maps aggregation modules.
        stages (nn.ModuleList): List of stage modules.
        final_agg (FeatureMapWeighting): Final feature maps aggregation module.
        head (nn.Sequential): Final 2D projection module.
        asp (Optional[AttentiveStatsPooling]): Attentive Statistics Pooling (ASTP) module.
        asp_bn (Optional[nn.BatchNorm1d]): Batch normalization module for ASTP.
        fc (nn.Module): Final linear transformation module.
    """

    def __init__(self, config: ReDimNet2Config):
        """
        Initialize a ReDimNet2 encoder.

        Args:
            config (ReDimNet2Config): Encoder configuration.

        Returns:
            None
        """
        super().__init__(config)

        self.pooling = config.pooling

        C = config.base_channels
        nb_freqs = config.mel_n_mels

        # 1D representations always have C*F channels as C and F evolve inversely.
        dim_1d = C * nb_freqs

        self.stem = nn.Sequential(
            nn.Conv2d(1, C, kernel_size=3, stride=1, padding="same"),
            LayerNorm(C),
            To1d(),
        )
        self.stem_norm = (
            nn.GroupNorm(C, dim_1d) if config.agg_group_norm else nn.Identity()
        )

        self.stage_aggs = nn.ModuleList()
        self.stages = nn.ModuleList()

        c, f = C, nb_freqs
        time_stride = 1

        for i, stage in enumerate(config.stages):
            assert f % stage.freq_stride == 0

            time_stride *= stage.time_stride

            self.stage_aggs.append(FeatureMapWeighting(i + 1, dim_1d))

            block_channels = int(stage.freq_stride * c * stage.channels_expansion)

            layers = [
                To2d(f, c),
                nn.Conv2d(
                    c,
                    block_channels,
                    kernel_size=(stage.freq_stride, time_stride),
                    stride=(stage.freq_stride, time_stride),
                    groups=(
                        math.gcd(c, block_channels) if config.compress_tconvs else 1
                    ),
                ),
            ]

            c = stage.freq_stride * c
            f = f // stage.freq_stride

            layers += [
                ResBasicBlock(
                    block_channels,
                    block_channels,
                    f,
                    se_channels=min(64, max(block_channels, 32)),
                    group_divisor=config.group_divisor,
                    use_fwSE=config.block_2d_type == Block2dTypeEnum.BASIC_RESNET_FWSE,
                )
                for _ in range(stage.nb_blocks)
            ]

            if block_channels != c:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(block_channels, c, kernel_size=1, padding="same"),
                        nn.BatchNorm2d(c, eps=1e-6),
                    )
                )

            layers.append(To1d())

            if stage.attention_reduction is not None:
                layers.append(
                    TimeContextBlock1d(
                        dim_1d,
                        dim_1d // stage.attention_reduction,
                        block_type=config.block_1d_type,
                    )
                )

            layers.append(TimeUpsample(time_stride))

            if config.agg_group_norm:
                layers.append(nn.GroupNorm(C, dim_1d))

            self.stages.append(nn.Sequential(*layers))

        self.time_stride = time_stride

        self.final_agg = FeatureMapWeighting(len(config.stages) + 1, dim_1d)

        out_size = dim_1d
        self.head = nn.Identity()
        if config.head_channels is not None:
            self.head = nn.Sequential(
                To2d(f, c),
                nn.Conv2d(c, config.head_channels, 1),
                nn.Flatten(start_dim=1, end_dim=2),
            )
            out_size = f * config.head_channels

        self.asp = None
        self.asp_bn = None
        if self.pooling:
            self.asp = AttentiveStatsPooling(
                out_size,
                attention_channels=config.attention_channels,
                global_context=config.global_context,
            )
            self.asp_bn = nn.BatchNorm1d(self.asp.out_size)

        self.fc = (
            nn.Linear(self.asp.out_size, self.encoder_dim)
            if self.pooling
            else nn.Conv1d(out_size, self.encoder_dim, kernel_size=1)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor. Shape: (N, L).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, D) or (N, D, L').
        """
        Z = super().forward(X)
        # Z: (N, F, T) = (N, 72, 200)

        # Number of frames must be a multiple of the total time stride.
        T = Z.size(-1)
        Z = Z[..., : (T // self.time_stride) * self.time_stride]

        Z = self.stem_norm(self.stem(Z.unsqueeze(1)))
        # Z: (N, C*F, T) = (N, 4608, 200)

        outputs = [Z]
        for agg, stage in zip(self.stage_aggs, self.stages):
            outputs.append(stage(agg(outputs)))

        Z = self.head(self.final_agg(outputs))
        # Z: (N, C', T) = (N, 2016, 200)

        if self.pooling:
            Z = self.asp_bn(self.asp(Z))

        return self.fc(Z)
