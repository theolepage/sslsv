import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .LIMLoss import LIMLoss, LIMLossEnum


@dataclass
class LIMConfig(BaseMethodConfig):

    loss_name: LIMLossEnum = LIMLossEnum.BCE


class LIM(BaseMethod):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.loss_fn = LIMLoss(config.loss_name)

    def forward(self, X, training=False):
        if not training: return self.encoder(X)
        
        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Y_1 = self.encoder(X_1)
        Y_2 = self.encoder(X_2)

        return Y_1, Y_2

    def train_step(self, Y, labels, step, samples):
        Y_1, Y_2 = Y
        
        N, _ = Y_1.size()

        shift = torch.randint(1, N, size=(1,)).item()
        Y_R = torch.roll(Y_2, shifts=shift, dims=0)

        pos = F.cosine_similarity(Y_1, Y_2, dim=-1)
        neg = F.cosine_similarity(Y_1, Y_R, dim=-1)

        loss = self.loss_fn(pos, neg)

        metrics = {
            'train/loss': loss,
        }

        return loss, metrics
