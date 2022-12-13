import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models.BaseModel import (
    BaseMomentumModel,
    BaseMomentumModelConfig,
    initialize_momentum_params
)


@dataclass
class BYOLConfig(BaseMomentumModelConfig):

    proj_hidden_dim: int = 4096
    proj_output_dim: int = 256

    pred_hidden_dim: int = 4096


class BYOL(BaseMomentumModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.proj_hidden_dim),
            nn.BatchNorm1d(config.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.proj_hidden_dim, config.proj_output_dim)
        )

        self.projector_momentum = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.proj_hidden_dim),
            nn.BatchNorm1d(config.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.proj_hidden_dim, config.proj_output_dim)
        )
        initialize_momentum_params(self.projector, self.projector_momentum)

        self.predictor = nn.Sequential(
            nn.Linear(config.proj_output_dim, config.pred_hidden_dim),
            nn.BatchNorm1d(config.pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.pred_hidden_dim, config.proj_output_dim)
        )

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

    def _byol_loss(self, P, Z):
        return 2 - 2 * F.cosine_similarity(P, Z.detach(), dim=-1).mean()

    def train_step(self, Z):
        Z_1, Z_2, P_1, P_2 = Z

        loss = self._byol_loss(P_1, Z_2) + self._byol_loss(P_2, Z_1)

        accuracy = InfoNCELoss.determine_accuracy(P_1, Z_2)
        accuracy += InfoNCELoss.determine_accuracy(P_2, Z_1)
        accuracy /= 2

        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy,
            'tau': self.momentum_updater.tau
        }

        return loss, metrics
