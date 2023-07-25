import math

import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.Custom import CustomLoss
from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models._BaseModel import BaseModel, BaseModelConfig


@dataclass
class CustomConfig(BaseModelConfig):
    
    loss_name: str = 'snt-xent'

    enable_multi_views: bool = False

    loss_margin_learnable: bool = False
    loss_margin: float = 0.2
    loss_scale: float = 5

    loss_simo: bool = False

    loss_reg_weight: float = 0.0

    enable_projector: bool = True
    projector_hidden_dim: int = 2048
    projector_output_dim: int = 256


class Custom(BaseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.epoch = 0
        self.max_epochs = 0

        self.config = config

        # self.projector = nn.Sequential(
        #     nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        # )

        self.loss_fn = CustomLoss(config)
    
    def _compute_embeddings(self, X):
        Y = self.encoder(X)

        if self.config.enable_projector:
            return self.projector(Y)

        return Y

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        # Retrieve global views
        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]
        views = [X_1, X_2]

        # Extract local views
        if self.config.enable_multi_views:
            small_frame_length = X.size(-1) // 2
            views.append(X_1[:, :small_frame_length])
            views.append(X_1[:, small_frame_length:])
            views.append(X_2[:, :small_frame_length])
            views.append(X_2[:, small_frame_length:])

        Z = torch.stack(
            [self._compute_embeddings(V) for V in views],
            dim=1
        )

        return Z

    def get_learnable_params(self):
        extra_learnable_params = [
            # {'params': self.projector.parameters()},
            {'params': self.loss_fn.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def on_train_epoch_start(self, epoch, max_epochs):
        self.epoch = epoch
        self.max_epochs = max_epochs

    def _loss_margin_scheduler(self):
        if self.config.loss_name != 'nt-xent-aam':
            return self.config.loss_margin

        if self.epoch > (self.max_epochs // 2):
            return self.config.loss_margin

        return (
            self.config.loss_margin -
            self.config.loss_margin *
            (math.cos(math.pi * self.epoch / (self.max_epochs // 2)) + 1) / 2
        )

    def train_step(self, Z, labels, step, samples):
        loss_margin = self._loss_margin_scheduler()
        self.loss_fn.loss_fn.margin = loss_margin

        loss = self.loss_fn(Z)

        metrics = {
            'train/loss': loss,
            # 'train/accuracy': InfoNCELoss.determine_accuracy(Z[:, 0], Z[:, 1]),
            'train/margin': loss_margin
        }

        return loss, metrics