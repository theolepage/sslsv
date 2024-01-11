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

import torch.distributed as dist
from sslsv.utils.distributed import gather, get_rank, get_world_size


@dataclass
class MoCoConfig(BaseMomentumModelConfig):

    start_tau: float = 0.999
    tau_scheduler: bool = False

    temperature: float = 0.2

    queue_size: int = 65536

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 256


class MoCo(BaseMomentumModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.epoch = 0

        self.queue_size = config.queue_size

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        )

        self.projector_momentum = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        )
        initialize_momentum_params(self.projector, self.projector_momentum)

        self.register_buffer(
            'queue',
            torch.randn(2, config.projector_output_dim, self.queue_size)
        )
        self.queue = F.normalize(self.queue, dim=1)

        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.loss_fn = MoCoLoss(config.temperature)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        x_gather = gather(x)

        idx_shuffle = torch.randperm(x_gather.size(0)).to(get_rank())
        dist.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)

        idx_this = idx_shuffle.view(get_world_size(), -1)[get_rank()]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        x_gather = gather(x)

        idx_this = idx_unshuffle.view(get_world_size(), -1)[get_rank()]

        return x_gather[idx_this]

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        # Queries
        Q_1 = self.projector(self.encoder(X[:, 0, :]))
        Q_2 = self.projector(self.encoder(X[:, 1, :]))

        # Keys
        X_s, idx_unshuffle = self._batch_shuffle_ddp(X)
        K_1 = self.projector_momentum(self.encoder_momentum(X_s[:, 0, :]))
        K_2 = self.projector_momentum(self.encoder_momentum(X_s[:, 1, :]))
        K_1 = self._batch_unshuffle_ddp(K_1, idx_unshuffle)
        K_2 = self._batch_unshuffle_ddp(K_2, idx_unshuffle)

        return Q_1, K_2, Q_2, K_1

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

    def on_train_epoch_start(self, epoch, max_epochs):
        self.epoch = epoch

    @torch.no_grad()
    def _enqueue(self, keys):
        batch_size = keys.size(1)

        assert self.queue_size % batch_size == 0

        ptr = int(self.queue_ptr)
        self.queue[:, :, ptr:ptr+batch_size] = keys.permute(0, 2, 1)

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def train_step(self, Z, labels, step, samples):
        Q_1, K_2, Q_2, K_1 = Z

        Q_1 = F.normalize(Q_1, p=2, dim=1)
        K_2 = F.normalize(K_2, p=2, dim=1)
        Q_2 = F.normalize(Q_2, p=2, dim=1)
        K_1 = F.normalize(K_1, p=2, dim=1)

        queue = self.queue.clone().detach()

        loss = self.loss_fn(Q_1, K_2, queue[1])
        loss += self.loss_fn(Q_2, K_1, queue[0])
        loss /= 2

        self._enqueue(torch.stack((
            gather(K_1),
            gather(K_2)
        )))

        metrics = {
            'train/loss': loss,
            # 'train/accuracy': InfoNCELoss.determine_accuracy(Q_1, Q_2),
            'train/tau': self.momentum_updater.tau
        }

        return loss, metrics