from dataclasses import dataclass
from typing import Callable

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from .BarlowTwinsLoss import BarlowTwinsLoss


@dataclass
class BarlowTwinsConfig(BaseSiameseMethodConfig):
    """
    Barlow Twins method configuration.

    Attributes:
        lamda (float): Redundancy reduction weight. Defaults to 0.005.
    """

    lamda: float = 0.005


class BarlowTwins(BaseSiameseMethod):
    """
    Barlow Twins method.

    Paper:
        Barlow Twins: Self-Supervised Learning via Redundancy Reduction
        *Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny*
        ICML 2021
        https://arxiv.org/abs/2103.03230

    Attributes:
        loss_fn (BarlowTwinsLoss): Loss function.
    """

    def __init__(
        self,
        config: BarlowTwinsConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a Barlow Twins method.

        Args:
            config (BarlowTwinsConfig): Method configuration.
            create_encoder_fn (Callable): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.loss_fn = BarlowTwinsLoss(config.lamda)