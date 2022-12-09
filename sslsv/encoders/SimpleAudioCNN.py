import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from dataclasses import dataclass

from sslsv.configs import EncoderConfig


@dataclass
class SimpleAudioCNNConfig(EncoderConfig):
    pass


class SimpleAudioCNNBlock(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.ln = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()

    def forward(self, X):
        return self.act(self.ln(self.conv(X)))


class SimpleAudioCNN(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder_dim = config.encoder_dim

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
                    paddings[i]
                )
            )
            last_dim = nb_filters[i]

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, X):
        X = X.unsqueeze(1) # (N, 1, L)
        Z = self.blocks(X)
        return Z