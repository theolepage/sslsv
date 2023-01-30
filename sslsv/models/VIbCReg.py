import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.modules.IterNorm import IterNorm
from sslsv.losses.VIbCReg import VIbCRegLoss
from sslsv.models._BaseSiameseModel import (
    BaseSiameseModel,
    BaseSiameseModelConfig
)


@dataclass
class VIbCRegConfig(BaseSiameseModelConfig):

    inv_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 8.0


class VIbCReg(BaseSiameseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            IterNorm(config.projector_output_dim, nb_groups=64, T=5, dim=2, affine=True)
        )

        self.loss_fn = VIbCRegLoss(
            config.inv_weight,
            config.var_weight,
            config.cov_weight
        )