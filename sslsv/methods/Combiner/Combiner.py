from dataclasses import dataclass
from enum import Enum
from typing import List

from sslsv.methods._BaseSiameseMethod import (
    BaseSiameseMethod,
    BaseSiameseMethodConfig
)

from sslsv.methods.CPC.InfoNCELoss import InfoNCELoss
from sslsv.methods.VICReg.VICRegLoss import VICRegLoss
from sslsv.methods.BarlowTwins.BarlowTwinsLoss import BarlowTwinsLoss


class LossTypeCombinerEnum(Enum):

    INFONCE     = 'infonce'
    VICREG      = 'vicreg'
    BARLOWTWINS = 'barlowtwins'


@dataclass
class LossItemCombinerConfig:

    type: LossTypeCombinerEnum
    weight: float = 1.0


@dataclass
class CombinerConfig(BaseSiameseMethodConfig):

    Y_losses: List[LossItemCombinerConfig] = None
    Z_losses: List[LossItemCombinerConfig] = None


class Combiner(BaseSiameseMethod):

    LOSS_FUNCTIONS = {
        LossTypeCombinerEnum.INFONCE     : InfoNCELoss(),
        LossTypeCombinerEnum.VICREG      : VICRegLoss(),
        LossTypeCombinerEnum.BARLOWTWINS : BarlowTwinsLoss()
    }

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Y_1 = self.encoder(X_1)
        Y_2 = self.encoder(X_2)

        Z_1 = self.projector(Y_1) if self.config.enable_projector else None
        Z_2 = self.projector(Y_2) if self.config.enable_projector else None

        return Y_1, Y_2, Z_1, Z_2

    def compute_loss(self, Z_1, Z_2, losses):
        loss = 0
        for l in losses:
            loss += l.weight * Combiner.LOSS_FUNCTIONS[l.type](Z_1, Z_2)
        return loss

    def train_step(self, Z, labels, step, samples):
        Y_1, Y_2, Z_1, Z_2 = Z

        loss = 0
        metrics = {}

        # Representations
        Y_loss = self.compute_loss(Y_1, Y_2, self.config.Y_losses)
        Y_accuracy = InfoNCELoss.determine_accuracy(Y_1, Y_2)
        metrics = {
            **metrics,
            'train/Y_loss': Y_loss,
            'train/Y_accuracy': Y_accuracy
        }
        loss += Y_loss

        # Embeddings
        if self.config.enable_projector:
            Z_loss = self.compute_loss(Z_1, Z_2, self.config.Z_losses)
            Z_accuracy = InfoNCELoss.determine_accuracy(Z_1, Z_2)
            metrics = {
                **metrics,
                'train/Z_loss': Z_loss,
                'train/Z_accuracy': Z_accuracy
            }
            loss += Z_loss

        metrics = {
            **metrics,
            'train/loss': loss
        }

        return loss, metrics
