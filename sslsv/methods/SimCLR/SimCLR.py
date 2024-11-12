from dataclasses import dataclass
from typing import Callable

from torch import nn

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from .SimCLRLoss import SimCLRLoss


@dataclass
class SimCLRConfig(BaseSiameseMethodConfig):
    """
    SimCLR method configuration.

    Attributes:
        temperature (float): Temperature value.
        projector_hidden_dim (int): Hhidden dimension of the projector network.
        projector_output_dim (int): Output dimension of the projector network.
    """

    temperature: float = 0.5

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 128


class SimCLR(BaseSiameseMethod):
    """
    SimCLR (SIMple framework for Contrastive Learning of visual Representations) method.

    Paper:
        A Simple Framework for Contrastive Learning of Visual Representations
        *Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton*
        ICML 2020
        https://arxiv.org/abs/2002.05709

    Attributes:
        projector (nn.Sequential): Projector module.
        loss_fn (SimCLRLoss): Loss function.
    """

    def __init__(
        self,
        config: SimCLRConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a SimCLR method.

        Args:
            config (SimCLRConfig): Method configuration.
            create_encoder_fn (Callable): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        if self.config.enable_projector:
            # self.projector = nn.Sequential(
            #     nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            # )
            self.projector = nn.Sequential(
                nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            )
            # self.projector = nn.Sequential(
            #     nn.Linear(self.encoder.encoder_dim, config.projector_output_dim),
            # )

        self.loss_fn = SimCLRLoss(config.temperature)
