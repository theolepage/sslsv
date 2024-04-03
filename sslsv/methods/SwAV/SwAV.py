import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .SinkhornKnopp import SinkhornKnopp
from .SwAVLoss import SwAVLoss

from sslsv.utils.distributed import get_world_size


@dataclass
class SwAVConfig(BaseMethodConfig):

    temperature: float = 0.1

    nb_prototypes: int = 3000

    sk_nb_iters: int = 3
    sk_epsilon: float = 0.05

    queue_size: int = 3840
    queue_start_epoch: int = 15

    freeze_prototypes_epochs: int = 1

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 128


class SwAV(BaseMethod):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.epoch = 0

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim)
        )

        self.prototypes = nn.utils.weight_norm(
            nn.Linear(
                config.projector_output_dim,
                config.nb_prototypes,
                bias=False
            )
        )
        self.prototypes.weight_g.data.fill_(1)
        self.prototypes.weight_g.requires_grad = False

        self.sk = SinkhornKnopp(
            nb_iters=config.sk_nb_iters,
            epsilon=config.sk_epsilon
        )

        self.loss_fn = SwAVLoss(config.temperature)

    def on_train_start(self, trainer):
        if self.config.queue_size > 0:
            self.register_buffer(
                'queue',
                torch.zeros(
                    2,
                    self.config.queue_size // get_world_size(),
                    self.config.projector_output_dim,
                    device=trainer.device,
                )
            )

    def forward(self, X, training=False):
        if not training: return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Z_1 = F.normalize(self.projector(self.encoder(X_1)), dim=-1)
        Z_2 = F.normalize(self.projector(self.encoder(X_2)), dim=-1)

        P_1 = self.prototypes(Z_1)
        P_2 = self.prototypes(Z_2)

        return Z_1, Z_2, P_1, P_2

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.projector.parameters()},
            {'params': self.prototypes.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def on_train_epoch_start(self, epoch, max_epochs):
        self.epoch = epoch

    def _get_sk_assignments(self, preds):
        N = preds[0].size(0)

        assignments = []
        
        use_queue = (
            self.config.queue_size > 0 and
            self.epoch >= self.config.queue_start_epoch
        )

        for i, P in enumerate(preds):
            if use_queue:
                P_queue = self.prototypes(self.queue[i])
                P = torch.cat((P, P_queue))
            assignments.append(self.sk(P)[:N])
        
        return assignments

    def train_step(self, Z, labels, step, samples):
        Z_1, Z_2, P_1, P_2 = Z

        N, _ = Z_1.size()

        preds = [P_1, P_2]
        assignments = self._get_sk_assignments(preds)

        loss = self.loss_fn(preds, assignments)

        # Update queue
        if self.config.queue_size > 0:
            self.queue[:, N:] = self.queue[:, :-N].clone()
            self.queue[0, :N] = Z_1.detach()
            self.queue[1, :N] = Z_2.detach()

        metrics = {
            'train/loss': loss
        }

        return loss, metrics

    def on_after_backward(self):
        if self.epoch < self.config.freeze_prototypes_epochs:
            for p in self.prototypes.parameters():
                p.grad = None