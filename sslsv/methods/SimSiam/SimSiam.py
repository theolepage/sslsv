import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .SimSiamLoss import SimSiamLoss


@dataclass
class SimSiamConfig(BaseMethodConfig):

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 2048

    pred_hidden_dim: int = 512


class SimSiam(BaseMethod):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim, bias=False),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_hidden_dim, bias=False),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            nn.BatchNorm1d(config.projector_output_dim, affine=False)
        )
        # hack: not use bias as it is followed by BN
        self.projector[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(config.projector_output_dim, config.pred_hidden_dim, bias=False),
            nn.BatchNorm1d(config.pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.pred_hidden_dim, config.projector_output_dim)
        )

        self.loss_fn = SimSiamLoss()

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

    def update_optim(
        self,
        optimizer,
        training_config,
        step,
        nb_steps,
        nb_steps_per_epoch
    ):
        lr = super().update_optim(
            optimizer,
            training_config,
            step,
            nb_steps,
            nb_steps_per_epoch
        )

        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = training_config.learning_rate

        return lr

    def train_step(self, Z, labels, step, samples):
        Z_1, Z_2, P_1, P_2 = Z

        loss = (
            self.loss_fn(P_1, Z_2) +
            self.loss_fn(P_2, Z_1)
        ) / 2

        metrics = {
            'train/loss': loss,
        }

        return loss, metrics
