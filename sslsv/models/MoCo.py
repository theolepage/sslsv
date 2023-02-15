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

    def forward(self, X, X_s=None, training=False):
        if not training: return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Q_1 = self.projector(self.encoder(X_1))
        Q_2 = self.projector(self.encoder(X_2))

        X_1_s = X_s[:, 0, :]
        X_2_s = X_s[:, 1, :]

        K_1 = self.projector_momentum(self.encoder_momentum(X_1_s))
        K_2 = self.projector_momentum(self.encoder_momentum(X_2_s))

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

        self._enqueue(torch.stack((K_1, K_2)))

        accuracy = InfoNCELoss.determine_accuracy(Q_1, Q_2)

        metrics = {
            'train/loss': loss,
            'train/accuracy': accuracy,
            'train/tau': self.momentum_updater.tau
        }

        return loss, metrics