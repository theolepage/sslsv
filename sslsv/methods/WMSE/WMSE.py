from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from sslsv.utils.distributed import gather

from .WMSELoss import WMSELoss
from .Whitening2d import Whitening2d


@dataclass
class WMSEConfig(BaseSiameseMethodConfig):

    projector_hidden_dim: int = 1024
    projector_output_dim: int = 64

    whitening_iters: int = 1
    whitening_size: int = 128
    whitening_eps: float = 0.0


class WMSE(BaseSiameseMethod):

    def __init__(
        self,
        config: WMSEConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
        )

        self.whitening = Whitening2d(
            config.projector_output_dim,
            eps=config.whitening_eps,
        )

        self.loss_fn = WMSELoss()

    def train_step(
        self,
        Z: Tuple[T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        Z_A = gather(Z[0])
        Z_B = gather(Z[1])

        N = Z_A.size(0)

        Z = torch.cat((Z_A, Z_B))

        loss = 0
        for _ in range(self.config.whitening_iters):
            z = torch.empty_like(Z)
            perm = torch.randperm(N).view(-1, self.config.whitening_size)
            for idx in perm:
                z[idx] = self.whitening(Z[idx])
                z[idx + N] = self.whitening(Z[idx + N])
            loss += self.loss_fn(z[:N], z[N : 2 * N])
        loss /= self.config.whitening_iters

        self.log_step_metrics(
            step,
            {
                "train/loss": loss,
            },
        )

        return loss
