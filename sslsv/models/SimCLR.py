import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models.BaseModel import BaseModel, BaseModelConfig


@dataclass
class SimCLRConfig(BaseModelConfig):
    
    enable_projector: bool = True
    projector_dim: int = 2048


class SimCLR(BaseModel):

    def __init__(self, config, encoder):
        super().__init__(config, encoder)

        self.enable_projector = config.enable_projector
        self.projector_dim = config.projector_dim

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoded_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim)
        )

        self.loss_fn = InfoNCELoss()

    def forward(self, X, training=False):
        Y = super().forward(X)

        if not training: return Y

        Z = self.projector(Y) if self.enable_projector else Y

        return Z

    def compute_loss(self, Z_1, Z_2):
        loss = self.loss_fn((Z_1, Z_2))

        accuracy = InfoNCELoss.determine_accuracy(Z_1, Z_2)
            
        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics
