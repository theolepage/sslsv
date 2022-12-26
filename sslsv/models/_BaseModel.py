import torch
import torch.nn as nn

from sslsv.encoders.ThinResNet34 import ThinResNet34

from dataclasses import dataclass

from sslsv.configs import ModelConfig


@dataclass
class BaseModelConfig(ModelConfig):
    pass


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