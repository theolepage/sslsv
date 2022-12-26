import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models._BaseSiameseModel import (
    BaseSiameseModel,
    BaseSiameseModelConfig
)


@dataclass
class SimCLRConfig(BaseSiameseModelConfig):
    
    temperature: float = 0.07

    enable_projector: bool = True
    projector_dim: int = 2048


class SimCLR(BaseSiameseModel):

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

        self.loss_fn = InfoNCELoss(config.temperature)