import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders.ThinResNet34 import ThinResNet34

from dataclasses import dataclass, field
from typing import List

from sslsv.configs import ModelConfig


@dataclass
class BaseModelConfig(ModelConfig):

    encoder_dim: int = 1024


class BaseModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder_dim = config.encoder_dim

        self.encoder = ThinResNet34(self.encoder_dim)

    def forward(self, X, training=False):
        return self.encoder(X)