import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.BarlowTwins import BarlowTwinsLoss
from sslsv.models.SimCLR import SimCLR, SimCLRConfig


@dataclass
class BarlowTwinsConfig(SimCLRConfig):

    lamda: float = 0.005


class BarlowTwins(SimCLR):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.loss_fn = BarlowTwinsLoss(config.lamda)