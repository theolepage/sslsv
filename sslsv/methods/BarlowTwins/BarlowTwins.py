from dataclasses import dataclass
from typing import Callable

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from .BarlowTwinsLoss import BarlowTwinsLoss


@dataclass
class BarlowTwinsConfig(BaseSiameseMethodConfig):

    lamda: float = 0.005


class BarlowTwins(BaseSiameseMethod):

    def __init__(
        self,
        config: BarlowTwinsConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        super().__init__(config, create_encoder_fn)

        self.loss_fn = BarlowTwinsLoss(config.lamda)
