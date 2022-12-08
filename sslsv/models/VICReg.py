import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.VICReg import VICRegLoss
from sslsv.models.SimCLR import SimCLR, SimCLRConfig


@dataclass
class VICRegConfig(SimCLRConfig):

    inv_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04


class VICReg(SimCLR):

    def __init__(self, config, encoder):
        super().__init__(config, encoder)

        self.loss_fn = VICRegLoss(
            config.inv_weight,
            config.var_weight,
            config.cov_weight
        )