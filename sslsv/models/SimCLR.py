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

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.enable_projector = config.enable_projector
        self.projector_dim = config.projector_dim

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim)
        )

        self.loss_fn = InfoNCELoss()

    def train_step(self, X):
        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Y_1 = self.forward(X_1)
        Y_2 = self.forward(X_2)

        Z_1 = self.projector(Y_1) if self.enable_projector else Y_1
        Z_2 = self.projector(Y_2) if self.enable_projector else Y_2

        loss = self.loss_fn((Z_1, Z_2))

        accuracy = InfoNCELoss.determine_accuracy(Z_1, Z_2)
            
        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics