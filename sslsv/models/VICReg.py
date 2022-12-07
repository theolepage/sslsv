import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass, field

from sslsv.losses.VICReg import VICRegLoss
from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models.BaseModel import BaseModel, BaseModelConfig


@dataclass
class VICRegConfig(BaseModelConfig):
    inv_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04


class VICReg(BaseModel):

    def __init__(self, config):
        super().__init__(config)

        self.loss_fn = VICRegLoss(
            config.inv_weight,
            config.var_weight,
            config.cov_weight
        )

    def compute_loss(self, Z_1, Z_2, Y_1, Y_2):
        loss = self.loss_fn((Z_1, Z_2))

        accuracy = InfoNCELoss.determine_accuracy(Z_1, Z_2)

        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics