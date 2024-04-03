import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig


@dataclass
class BaseSiameseMethodConfig(BaseMethodConfig):
    
    enable_projector: bool = True

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 2048


class BaseSiameseMethod(BaseMethod):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        if config.enable_projector:
            self.projector = nn.Sequential(
                nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
            )

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Y_1 = self.encoder(X_1)
        Y_2 = self.encoder(X_2)

        Z_1 = self.projector(Y_1) if self.config.enable_projector else Y_1
        Z_2 = self.projector(Y_2) if self.config.enable_projector else Y_2

        return Z_1, Z_2

    def get_learnable_params(self):
        if self.config.enable_projector:
            return super().get_learnable_params() + [
                {'params': self.projector.parameters()}
            ]
        return super().get_learnable_params()

    def train_step(self, Z, labels, step, samples):
        Z_1, Z_2 = Z

        loss = self.loss_fn(Z_1, Z_2)

        metrics = {
            'train/loss': loss,
        }

        return loss, metrics