import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders.ThinResNet34 import ThinResNet34

from dataclasses import dataclass, field
from typing import List

from sslsv.configs import ModelConfig


@dataclass
class BaseModelConfig(ModelConfig):

    # Projector
    enable_projector: bool = True
    projector_dim: int = 2048


class BaseModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.enable_projector = config.enable_projector
        self.projector_dim = config.projector_dim

        self.encoder = ThinResNet34()
        self.projector = nn.Sequential(
            nn.Linear(1024, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim)
        )

    def forward(self, X, training=False):
        Y = self.encoder(X)

        # Do not use projector for inference / evaluaton
        if not training: return Y 

        Z = self.projector(Y) if self.enable_projector else Y
        
        return Y, Z
