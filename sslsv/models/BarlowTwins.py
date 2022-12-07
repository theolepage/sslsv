import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass, field

from sslsv.losses.BarlowTwins import BarlowTwinsLoss
from sslsv.models.BaseModel import BaseModel, BaseModelConfig


@dataclass
class BarlowTwinsConfig(BaseModelConfig):
    lamda: float = 0.005


class BarlowTwins(BaseModel):

    def __init__(self, config):
        super().__init__(config)

        self.loss_fn = BarlowTwinsLoss(config.lamda)

    def compute_loss(self, Z_1, Z_2, Y_1, Y_2):
        loss = self.loss_fn((Z_1, Z_2))

        accuracy = InfoNCELoss.determine_accuracy(Z_1, Z_2)
            
        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics
