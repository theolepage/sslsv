import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.VICReg import VICRegLoss
from sslsv.models._BaseSiameseModel import (
    BaseSiameseModel,
    BaseSiameseModelConfig
)


@dataclass
class VICRegConfig(BaseSiameseModelConfig):

    inv_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04


class VICReg(BaseSiameseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.loss_fn = VICRegLoss(
            config.inv_weight,
            config.var_weight,
            config.cov_weight
        )