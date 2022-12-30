import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.losses.DINO import DINOLoss
from sslsv.models._BaseMomentumModel import (
    BaseMomentumModel,
    BaseMomentumModelConfig,
    initialize_momentum_params
)


class DINOHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, bottleneck_dim, output_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
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
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)
        return x


@dataclass
class DINOConfig(BaseMomentumModelConfig):

    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_output_dim: int = 65536

    clip_grad: bool = False
    clip_grad_max: float = 3.0

    freeze_last_layer: int = 1

    student_temperature: float = 0.1
    teacher_temperature: float = 0.07
    teacher_temperature_warmup: float = 0.04
    teacher_temperature_warmup_epochs: float = 30

    small_frame_length: int = 32000


class DINO(BaseMomentumModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.current_epoch = 0

        self.clip_grad = config.clip_grad
        self.clip_grad_max = config.clip_grad_max
        self.freeze_last_layer = config.freeze_last_layer
        self.small_frame_length = config.small_frame_length

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
        if not training: return self.encoder(X)

        # Retrieve global views
        X_1_large = X[:, 0, :]
        X_2_large = X[:, 1, :]

        # Extract local views
        X_1_small_A = X_1_large[:, :self.small_frame_length]
        X_1_small_B = X_1_large[:, self.small_frame_length:]
        X_2_small_A = X_2_large[:, :self.small_frame_length]
        X_2_small_B = X_2_large[:, self.small_frame_length:]

        # Compute embeddings
        S = torch.cat((
            self.head(self.encoder(X_1_small_A)),
            self.head(self.encoder(X_1_small_B)),
            self.head(self.encoder(X_2_small_A)),
            self.head(self.encoder(X_2_small_B))
        ))

        T = torch.cat((
            self.head_momentum(self.encoder_momentum(X_1_large)),
            self.head_momentum(self.encoder_momentum(X_2_large))
        ))

        return S, T

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.head.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

    def get_momentum_pairs(self):
        extra_momentum_pairs = [
            (self.head, self.head_momentum)
        ]
        return super().get_momentum_pairs() + extra_momentum_pairs

    def train_step(self, Z):
        S, T = Z

        loss = self.loss_fn(S, T)

        metrics = {
            'train_loss': loss,
            'tau': self.momentum_updater.tau
        }

        return loss, metrics

    def on_train_epoch_start(self, epoch, max_epochs):
        self.current_epoch = epoch
        self.loss_fn.epoch = epoch

    def on_after_backward(self):
        # Perform gradient clipping on encoder
        if self.clip_grad:
            for p in self.encoder.parameters():
                if p.grad is not None:
                    clip = self.clip_grad_max / (p.grad.data.norm(2) + 1e-6)
                    if clip < 1:
                        p.grad.data.mul_(clip)

        # Freeze last layer of head
        if self.current_epoch < self.freeze_last_layer:
            for p in self.head.last_layer.parameters():
                p.grad = None