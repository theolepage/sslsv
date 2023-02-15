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

    loss_margin_learnable: bool = False
    loss_margin: float = 0.2
    loss_scale: float = 5

    loss_reg_weight: float = 0.0

    enable_projector: bool = True
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
            {'params': self.projector.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def train_step(self, Z, labels, step, samples):
        loss = self.loss_fn(Z)

        accuracy = InfoNCELoss.determine_accuracy(Z[:, 0], Z[:, 1])

        metrics = {
            'train/loss': loss,
            'train/accuracy': accuracy
        }

        return loss, metrics