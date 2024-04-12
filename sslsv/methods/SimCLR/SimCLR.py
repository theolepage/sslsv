from torch import nn

from dataclasses import dataclass

from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from .SimCLRLoss import SimCLRLoss


@dataclass
class SimCLRConfig(BaseSiameseMethodConfig):

    temperature: float = 0.2

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 256


class SimCLR(BaseSiameseMethod):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        if self.config.enable_projector:
            self.projector = nn.Sequential(
                nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            )

        self.loss_fn = SimCLRLoss(config.temperature)
