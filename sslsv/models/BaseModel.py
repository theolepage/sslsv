import torch
from torch import nn
import torch.nn.functional as F

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

    def forward(self, X):
        return self.encoder(X)

    def train_step(self, X):
        raise NotImplementedError

    def on_train_step_start(self):
        pass

    def on_train_step_end(self):
        pass