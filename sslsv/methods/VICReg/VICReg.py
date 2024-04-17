from dataclasses import dataclass
from typing import Callable

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from .VICRegLoss import VICRegLoss


@dataclass
class VICRegConfig(BaseSiameseMethodConfig):

    inv_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04


class VICReg(BaseSiameseMethod):

    def __init__(
        self,
        config: VICRegConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        super().__init__(config, create_encoder_fn)

        self.loss_fn = VICRegLoss(
            config.inv_weight,
            config.var_weight,
            config.cov_weight,
        )
