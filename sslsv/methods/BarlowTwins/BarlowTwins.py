import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.methods._BaseSiameseMethod import (
    BaseSiameseMethod,
    BaseSiameseMethodConfig
)

from .BarlowTwinsLoss import BarlowTwinsLoss


@dataclass
class BarlowTwinsConfig(BaseSiameseMethodConfig):

    lamda: float = 0.005


class BarlowTwins(BaseSiameseMethod):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.loss_fn = BarlowTwinsLoss(config.lamda)