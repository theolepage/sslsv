import math

import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders.ThinResNet34 import ThinResNet34

from dataclasses import dataclass

from sslsv.configs import ModelConfig


@torch.no_grad()
def initialize_momentum_params(model, model_momentum):
    for p, p_m in zip(model.parameters(), model_momentum.parameters()):
        p_m.data.copy_(p.data)
        p_m.requires_grad = False


class MomentumUpdater:

    def __init__(self, start_tau, end_tau):
        super().__init__()

        assert 0 <= start_tau <= 1
        assert 0 <= end_tau <= 1 and start_tau <= end_tau

        self.start_tau = start_tau
        self.end_tau = end_tau

        self.tau = start_tau

    @torch.no_grad()
    def update(self, model, model_momentum):
        for p, p_m in zip(model.parameters(), model_momentum.parameters()):
            p_m.data = self.tau * p_m.data + (1 - self.tau) * p.data

    def update_tau(self, step, max_steps):
        self.tau = (
            self.end_tau -
            (self.end_tau - self.start_tau) *
            (math.cos(math.pi * step / max_steps) + 1) / 2
        )


@dataclass
class BaseModelConfig(ModelConfig):
    pass


@dataclass
class BaseMomentumModelConfig(BaseModelConfig):
    
    tau_scheduler: bool = True
    start_tau: float = 0.99
    end_tau: float = 1.0


class BaseModel(nn.Module):

    def __init__(self, config, create_encoder_fn):
        super().__init__()

        self.encoder = create_encoder_fn()

    def forward(self, X, training=False):
        return self.encoder(X)

    def get_learnable_params(self):
        return [{'params': self.encoder.parameters()}]

    def get_initial_learning_rate(self, training_config):
        return training_config.learning_rate

    def adjust_learning_rate(self, optimizer, learning_rate, epoch, epochs):
        # Equivalent to StepLR(..., step_size=10, gamma=0.95)
        lr = learning_rate * (0.95 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_step(self, X):
        raise NotImplementedError

    def on_train_step_start(self, step, max_steps):
        pass

    def on_train_step_end(self, step, max_steps):
        pass


class BaseMomentumModel(BaseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.encoder_momentum = create_encoder_fn()
        initialize_momentum_params(self.encoder, self.encoder_momentum)

        self.tau_scheduler = config.tau_scheduler
        self.momentum_updater = MomentumUpdater(
            config.start_tau,
            config.end_tau
        )

    def get_momentum_pairs(self):
        return [(self.encoder, self.encoder_momentum)]

    def on_train_step_end(self, step, max_steps):
        for m, mm in self.get_momentum_pairs():
            self.momentum_updater.update(m, mm)

        if self.tau_scheduler:
            self.momentum_updater.update_tau(step, max_steps)