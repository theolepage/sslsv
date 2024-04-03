import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass

from sslsv.methods._BaseMomentumMethod import (
    BaseMomentumMethod,
    BaseMomentumMethodConfig,
    initialize_momentum_params
)

from .DINOLoss import DINOLoss


class DINOHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, bottleneck_dim, output_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)
        return x


@dataclass
class DINOConfig(BaseMomentumMethodConfig):

    start_tau: float = 0.996

    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_output_dim: int = 65536

    freeze_last_layer: int = 1

    student_temperature: float = 0.1
    teacher_temperature: float = 0.04
    teacher_temperature_warmup: float = 0.04
    teacher_temperature_warmup_epochs: float = 10


class DINO(BaseMomentumMethod):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.current_epoch = 0

        self.freeze_last_layer = config.freeze_last_layer

        self.head = DINOHead(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            output_dim=config.head_output_dim
        )

        self.head_momentum = DINOHead(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            output_dim=config.head_output_dim
        )
        initialize_momentum_params(self.head, self.head_momentum)

        self.loss_fn = DINOLoss(
            nb_prototypes=config.head_output_dim,
            student_temp=config.student_temperature,
            teacher_temp=config.teacher_temperature,
            teacher_temp_warmup=config.teacher_temperature_warmup,
            teacher_temp_warmup_epochs=config.teacher_temperature_warmup_epochs
        )

    def forward(self, X, training=False):
        if not training: return self.encoder_momentum(X)

        N, V, L = X.shape

        X = X.transpose(0, 1)

        global_frames = X[:2, :, :].reshape(-1, L)
        local_frames = X[2:, :, :L // 2].reshape(-1, L // 2)

        T = self.head_momentum(self.encoder_momentum(global_frames))

        S = torch.cat((
            self.head(self.encoder(global_frames)),
            self.head(self.encoder(local_frames))
        ), axis=0)

        return S, T

    def update_optim(
        self,
        optimizer,
        training_config,
        step,
        nb_steps,
        nb_steps_per_epoch
    ):
        init_lr = training_config.learning_rate
        wd = training_config.weight_decay

        lr_schedule = (
            1e-4 + 0.5 * (init_lr - 1e-4) *
            (1 + np.cos(np.pi * np.arange(nb_steps) / nb_steps))
        )
        lr = lr_schedule[step]
        
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
            param_group['weight_decay'] = wd if i == 0 else 0
        return lr

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.head.parameters()}
        ]
        params = super().get_learnable_params() + extra_learnable_params

        # Do not apply weight decay on biases and norms parameters
        regularized = []
        not_regularized = []
        for module in params:
            for param in module['params']:
                if not param.requires_grad: continue

                if len(param.shape) == 1:
                    not_regularized.append(param)
                else:
                    regularized.append(param)
        
        return [{'params': regularized}, {'params': not_regularized}]

    def get_momentum_pairs(self):
        extra_momentum_pairs = [
            (self.head, self.head_momentum)
        ]
        return super().get_momentum_pairs() + extra_momentum_pairs

    def train_step(self, Z, labels, step, samples):
        S, T = Z

        loss = self.loss_fn(S, T)

        metrics = {
            'train/loss': loss,
            'train/tau': self.momentum_updater.tau
        }

        return loss, metrics

    def on_train_epoch_start(self, epoch, max_epochs):
        self.current_epoch = epoch
        self.loss_fn.epoch = epoch

    def on_after_backward(self):
        # Freeze last layer of head
        if self.current_epoch < self.freeze_last_layer:
            for p in self.head.last_layer.parameters():
                p.grad = None