import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.methods._BaseSiameseMethod import (
    BaseSiameseMethod,
    BaseSiameseMethodConfig
)

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

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        )

        self.whitening = Whitening2d(
            config.projector_output_dim,
            eps=config.whitening_eps
        )

        self.loss_fn = WMSELoss()


    def train_step(self, Z, labels, step, samples):
        from sslsv.utils.distributed import gather
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
            loss += self.loss_fn(z[:N], z[N:2*N])
        loss /= self.config.whitening_iters

        metrics = {
            'train/loss': loss,
        }

        return loss, metrics