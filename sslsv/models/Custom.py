import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.Custom import CustomLoss
from sslsv.models._BaseSiameseModel import (
    BaseSiameseModel,
    BaseSiameseModelConfig
)


@dataclass
class CustomConfig(BaseSiameseModelConfig):
    
    loss_name: str = 'contrastive'

    loss_learnable_hyperparams: bool = False
    loss_margin: float = 0.2
    loss_scale: float = 30
    loss_temperature: float = 0.2
    loss_init_w: float = 10
    loss_init_b: float = -5
    loss_vicreg_scale: float = 1.0
    loss_vicreg_inv_weight: float = 1.0
    loss_vicreg_var_weight: float = 1.0
    loss_vicreg_cov_weight: float = 0.04

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 256


class Custom(BaseSiameseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        )

        self.loss_fn = CustomLoss(config)