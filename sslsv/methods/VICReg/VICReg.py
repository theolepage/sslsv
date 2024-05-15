from dataclasses import dataclass
from typing import Callable

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from .VICRegLoss import VICRegLoss


@dataclass
class VICRegConfig(BaseSiameseMethodConfig):
    """
    VICReg method configuration.

    Attributes:
        inv_weight (float): Weight of invariance loss term.
        var_weight (float): Weight of variance loss term.
        cov_weight (float): Weight of covariance loss term.
    """

    inv_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04


class VICReg(BaseSiameseMethod):
    """
    VICReg (Variance-Invariance-Covariance Regularization) method.

    Paper:
        VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
        *Adrien Bardes, Jean Ponce, Yann LeCun*
        ICLR 2022
        https://arxiv.org/abs/2105.04906

    Attributes:
        loss_fn (VICRegLoss): Loss function.
    """

    def __init__(
        self,
        config: VICRegConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a VICReg method.

        Args:
            config (VICRegConfig): Method configuration.
            create_encoder_fn (Callable): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.loss_fn = VICRegLoss(
            config.inv_weight,
            config.var_weight,
            config.cov_weight,
        )
