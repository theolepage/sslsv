import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.modules.IterNorm import IterNorm
from sslsv.losses.VIbCReg import VIbCRegLoss
from sslsv.models.SimCLR import SimCLR, SimCLRConfig


@dataclass
class VIbCRegConfig(SimCLRConfig):

    inv_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 8.0


class VIbCReg(SimCLR):

    def __init__(self, config, encoder):
        super().__init__(config, encoder)

        self.projector = nn.Sequential(
            nn.Linear(1024, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim),
            IterNorm(self.projector_dim, nb_groups=64, T=5, dim=2, affine=True)
        )

        self.loss_fn = VIbCRegLoss(
            config.inv_weight,
            config.var_weight,
            config.cov_weight
        )