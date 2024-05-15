from dataclasses import dataclass
from typing import Callable

from torch import nn

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from .IterNorm import IterNorm
from .VIbCRegLoss import VIbCRegLoss


@dataclass
class VIbCRegConfig(BaseSiameseMethodConfig):
    """
    VIbCReg method configuration.

    Attributes:
        inv_weight (float): Weight of invariance loss term.
        var_weight (float): Weight of variance loss term.
        cov_weight (float): Weight of covariance loss term.
    """

    inv_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 8.0


class VIbCReg(BaseSiameseMethod):
    """
    VIbCReg (Variance-Invariance-better-Covariance Regularization) method.

    Paper:
        Computer Vision Self-supervised Learning Methods on Time Series
        *Daesoo Lee, Erlend Aune*
        arXiv preprint 2022
        https://arxiv.org/abs/2109.00783

    Attributes:
        projector (nn.Sequential): Projection module.
        loss_fn (VIbCRegLoss): Loss function.
    """

    def __init__(
        self,
        config: VIbCRegConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a VIbCReg method.

        Args:
            config (VIbCRegConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            IterNorm(
                config.projector_output_dim,
                nb_groups=64,
                T=5,
                dim=2,
                affine=True,
            ),
        )

        self.loss_fn = VIbCRegLoss(
            config.inv_weight,
            config.var_weight,
            config.cov_weight,
        )
