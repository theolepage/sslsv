import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List

from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.losses.VICReg import VICRegLoss
from sslsv.losses.BarlowTwins import BarlowTwinsLoss
from sslsv.models.SimCLR import SimCLR, SimCLRConfig
from sslsv.configs import ModelConfig


@dataclass
class ElementMultiLossesConfig:

    name: str = None
    weight: float = 1.0


@dataclass
class MultiLossesConfig(SimCLRConfig):

    Y_losses: List[ElementMultiLossesConfig] = None
    Z_losses: List[ElementMultiLossesConfig] = None


class MultiLosses(SimCLR):

    LOSS_FUNCTIONS = {
        'infonce':     InfoNCELoss(),
        'vicreg':      VICRegLoss(),
        'barlowtwins': BarlowTwinsLoss()
    }

    def __init__(self, config, encoder):
        super().__init__(config, encoder)

        self.config = config

    def compute_loss_(self, Z_1, Z_2, losses):
        loss = 0
        for l in losses:
            loss += l.weight * MultiLosses.LOSS_FUNCTIONS[l.name]((Z_1, Z_2))
        return loss

    def compute_loss(self, Z_1, Z_2):
        Y_1, Z_1 = Z_1
        Y_2, Z_2 = Z_2

        loss = 0
        metrics = {}

        # Representations
        Y_loss = self.compute_loss_(Y_1, Y_2, self.config.Y_losses)
        Y_accuracy = InfoNCELoss.determine_accuracy(Y_1, Y_2)
        metrics = {
            **metrics,
            'train_Y_loss': Y_loss,
            'train_Y_accuracy': Y_accuracy
        }
        loss += Y_loss

        # Embeddings
        if self.enable_projector:
            Z_loss = self.compute_loss_(Z_1, Z_2, self.config.Z_losses)
            Z_accuracy = InfoNCELoss.determine_accuracy(Z_1, Z_2)
            metrics = {
                **metrics,
                'train_Z_loss': Z_loss,
                'train_Z_accuracy': Z_accuracy
            }
            loss += Z_loss

        metrics = {
            **metrics,
            'train_loss': loss
        }

        return loss, metrics
