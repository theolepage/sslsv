import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.Custom import CustomLoss
from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models._BaseModel import BaseModel, BaseModelConfig


@dataclass
class CustomConfig(BaseModelConfig):
    
    loss_name: str = 'nsoftmax'

    enable_multi_views: bool = False

    loss_learnable_hyperparams: bool = False
    loss_margin: float = 0.2
    loss_scale: float = 5
    loss_init_w: float = 10
    loss_init_b: float = -5
    loss_vicreg_scale: float = 1.0
    loss_vicreg_inv_weight: float = 1.0
    loss_vicreg_var_weight: float = 1.0
    loss_vicreg_cov_weight: float = 0.04

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 256


class Custom(BaseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.config = config

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        )

        self.loss_fn = CustomLoss(config)

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        # Retrieve global views
        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]
        views = [X_1, X_2]

        # Extract local views
        if self.config.enable_multi_views:
            small_frame_length = X.size(-1) // 2
            views += X_1[:, :small_frame_length]
            views += X_1[:, small_frame_length:]
            views += X_2[:, :small_frame_length]
            views += X_2[:, small_frame_length:]

        Z = torch.stack(
            [self.projector(self.encoder(V)) for V in views],
            dim=1
        )

        return Z

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.projector.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def train_step(self, Z, labels, step, samples):
        loss = self.loss_fn(Z)

        accuracy = InfoNCELoss.determine_accuracy(Z[:, 0], Z[:, 1])

        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics