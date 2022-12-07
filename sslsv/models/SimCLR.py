import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass, field

from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models.BaseModel import BaseModel, BaseModelConfig


@dataclass
class SimCLRConfig(BaseModelConfig):
    pass


class SimCLR(BaseModel):

    def __init__(self, config):
        super().__init__(config)

        self.loss_fn = InfoNCELoss()

    def compute_loss(self, Z_1, Z_2, Y_1, Y_2):
        loss = self.loss_fn((Z_1, Z_2))

        accuracy = InfoNCELoss.determine_accuracy(Z_1, Z_2)
            
        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics
