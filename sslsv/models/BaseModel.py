import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders.ThinResNet34 import ThinResNet34

from dataclasses import dataclass

from sslsv.configs import ModelConfig


@dataclass
class BaseModelConfig(ModelConfig):
    pass


class BaseModel(nn.Module):

    def __init__(self, config, encoder):
        super().__init__()

        self.encoder = encoder

    def forward(self, X, training=False):
        return self.encoder(X)