import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.methods._BaseMomentumMethod import (
    BaseMomentumMethod,
    BaseMomentumMethodConfig,
    initialize_momentum_params
)

from .BYOLLoss import BYOLLoss


@dataclass
class BYOLConfig(BaseMomentumMethodConfig):

    projector_hidden_dim: int = 4096
    projector_output_dim: int = 256

    predictor_hidden_dim: int = 4096


class BYOL(BaseMomentumMethod):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        )

        self.projector_momentum = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        )
        initialize_momentum_params(self.projector, self.projector_momentum)

        self.predictor = nn.Sequential(
            nn.Linear(config.projector_output_dim, config.predictor_hidden_dim),
            nn.BatchNorm1d(config.predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.predictor_hidden_dim, config.projector_output_dim)
        )

        self.loss_fn = BYOLLoss()

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        P_1 = self.predictor(self.projector(self.encoder(X_1)))
        Z_1 = self.projector_momentum(self.encoder_momentum(X_1))

        P_2 = self.predictor(self.projector(self.encoder(X_2)))
        Z_2 = self.projector_momentum(self.encoder_momentum(X_2))

        return Z_1, Z_2, P_1, P_2

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.projector.parameters()},
            {'params': self.predictor.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def get_momentum_pairs(self):
        extra_momentum_pairs = [
            (self.projector, self.projector_momentum)
        ]
        return super().get_momentum_pairs() + extra_momentum_pairs

    def train_step(self, Z, labels, step, samples):
        Z_1, Z_2, P_1, P_2 = Z

        loss = self.loss_fn(P_1, Z_2) + self.loss_fn(P_2, Z_1)

        metrics = {
            'train/loss': loss,
            'train/tau': self.momentum_updater.tau
        }

        return loss, metrics
