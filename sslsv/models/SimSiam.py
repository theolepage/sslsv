import math

import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models.BaseModel import BaseModel, BaseModelConfig


@dataclass
class SimSiamConfig(BaseModelConfig):

    proj_hidden_dim: int = 2048
    proj_output_dim: int = 2048

    pred_hidden_dim: int = 512


class SimSiam(BaseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.proj_hidden_dim, bias=False),
            nn.BatchNorm1d(config.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.proj_hidden_dim, config.proj_hidden_dim, bias=False),
            nn.BatchNorm1d(config.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.proj_hidden_dim, config.proj_output_dim),
            nn.BatchNorm1d(config.proj_output_dim, affine=False)
        )
        # hack: not use bias as it is followed by BN
        self.projector[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(config.proj_output_dim, config.pred_hidden_dim, bias=False),
            nn.BatchNorm1d(config.pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.pred_hidden_dim, config.proj_output_dim)
        )

    def forward(self, X, training=False):
        if not training: return self.encoder(X)
        
        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Z_1 = self.projector(self.encoder(X_1))
        P_1 = self.predictor(Z_1)

        Z_2 = self.projector(self.encoder(X_2))
        P_2 = self.predictor(Z_2)

        return Z_1, Z_2, P_1, P_2

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.projector.parameters()},
            {'params': self.predictor.parameters(), 'fix_lr': True}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def get_initial_learning_rate(self, training_config):
        return training_config.learning_rate * training_config.batch_size / 256

    def adjust_learning_rate(self, optimizer, learning_rate, epoch, epochs):
        lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = learning_rate
            else:
                param_group['lr'] = lr
        return lr

    def _simsiam_loss(self, P, Z):
        return -F.cosine_similarity(P, Z.detach(), dim=-1).mean()

    def train_step(self, Z):
        Z_1, Z_2, P_1, P_2 = Z

        loss = (
            self._simsiam_loss(P_1, Z_2) + self._simsiam_loss(P_2, Z_1)
        ) / 2

        accuracy = InfoNCELoss.determine_accuracy(Z_1, Z_2)

        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics
