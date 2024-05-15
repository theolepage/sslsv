from dataclasses import dataclass

import torch
from torch import nn

from sslsv.encoders._BaseEncoder import BaseEncoder, BaseEncoderConfig


@dataclass
class SimpleAudioCNNConfig(BaseEncoderConfig):
    """
    SimpleAudioCNN encoder configuration.

    Attributes:
        extract_mel_features (bool): Whether to extract Mel Spectrogram features.
    """

    extract_mel_features: bool = False


class SimpleAudioCNNBlock(nn.Module):
    """
    SimpleAudioCNN main block.

    Attributes:
        conv (nn.Conv1d): Convolutional layer.
        ln (nn.BatchNorm1d): Batch normalization layer.
        act (nn.ReLU): Activation function.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        """
        Initialize a SimpleAudioCNNBlock module.

        Args:
            in_dim (int): Number of input channels.
            out_dim (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convoluton.
            padding (int): Padding of the convolution.

        Returns:
            None
        """
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
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.act(self.ln(self.conv(X)))


class SimpleAudioCNN(BaseEncoder):
    """
    SimpleAudioCNN encoder.

    Paper:
        Representation Learning with Contrastive Predictive Coding
        *Aaron van den Oord, Yazhe Li, Oriol Vinyals*
        arXiv preprint 2019
        https://arxiv.org/abs/1807.03748

    Attributes:
        nb_filters (List[int]): Nnumber of filters for each convolutional block.
        kernel_sizes (List[int]): Kernel size for each convolutional block.
        strides (List[int]): Stride for each convolutional block.
        paddings (List[int]): Padding for each convolutional block.
        blocks (nn.Sequential): Sequential module of convolutional blocks.
    """

    def __init__(self, config: SimpleAudioCNNConfig):
        """
        Initialize a SimpleAudioCNN encoder.

        Args:
            config (SimpleAudioCNNConfig): Encoder configuration.

        Returns:
            None
        """
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
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        Z = super().forward(X)

        Z = Z.unsqueeze(1)
        # Z: (N, 1, L)

        Z = self.blocks(Z)

        return Z
