import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.MoCo import MoCoLoss
from sslsv.losses.InfoNCE import InfoNCELoss
from sslsv.models._BaseMomentumModel import (
    BaseMomentumModel,
    BaseMomentumModelConfig,
    initialize_momentum_params
)


@dataclass
class MoCoConfig(BaseMomentumModelConfig):
    
    end_tau: float = 0.999

    temperature: float = 0.07

    queue_size: int = 65536

    projector_dim: int = 2048


class MoCo(BaseMomentumModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.temperature = config.temperature
        self.queue_size = config.queue_size
        self.projector_dim = config.projector_dim

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim)
        )

        self.projector_momentum = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.projector_dim)
        )
        initialize_momentum_params(self.projector, self.projector_momentum)

        self.register_buffer(
            'queue',
            torch.randn(self.projector_dim, self.queue_size)
        )
        self.queue = F.normalize(self.queue, dim=1)

        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.loss_fn = MoCoLoss(config.temperature)

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Q = self.projector(self.encoder(X_1))
        K = self.projector_momentum(self.encoder_momentum(X_2))

        return Q, K

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.projector.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def get_momentum_pairs(self):
        extra_momentum_pairs = [
            (self.projector, self.projector_momentum)
        ]
        return super().get_momentum_pairs() + extra_momentum_pairs

    @torch.no_grad()
    def _enqueue(self, keys):
        batch_size = keys.size(0)

        assert self.queue_size % batch_size == 0

        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr+batch_size] = keys.T

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def train_step(self, Z):
        Q, K = Z

        Q = F.normalize(Q, p=2, dim=1)
        K = F.normalize(K, p=2, dim=1)

        loss = self.loss_fn(Q, K, self.queue.clone().detach())

        self._enqueue(K)

        accuracy = InfoNCELoss.determine_accuracy(Q, K)

        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics