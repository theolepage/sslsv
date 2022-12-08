import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
from enum import Enum

from sslsv.models.BaseModel import BaseModel, BaseModelConfig


class LIMLossFnEnum(Enum):

    BCE = 'bce'
    NCE = 'nce'
    MINE = 'mine'


@dataclass
class LIMConfig(BaseModelConfig):

    loss_fn: LIMLossFnEnum = 'bce'
    context_length: int = 1


class LIM(BaseModel):

    def __init__(self, config, encoder):
        super().__init__(config, encoder)

        self.loss_fn = config.loss_fn
        self.context_length = config.context_length

        self.discriminator = nn.Sequential(
            nn.Linear(2 * self.encoder.encoded_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, X, training=False):
        Y = super().forward(X)
        return Y if training else Y.mean(dim=2)

    def _extract_chunks(self, Y):
        N, C, L = Y.size()

        max_idx = L - self.context_length + 1
        idx1, idx2, idx3 = torch.randint(0, max_idx, size=(3,))
        shift = torch.randint(1, N, size=(1,)).item()

        C1 = Y[:, :, idx1:idx1+self.context_length]
        C1 = torch.mean(C1, dim=2)

        C2 = Y[:, :, idx2:idx2+self.context_length]
        C2 = torch.mean(C2, dim=2)

        CR = Y[:, :, idx3:idx3+self.context_length]
        CR = torch.mean(CR, dim=2)
        CR = torch.roll(CR, shifts=shift, dims=0)

        return C1, C2, CR

    def _bce_loss(self, pos, neg, eps=1e-07):
        pos = torch.clamp(torch.sigmoid(pos), eps, 1.0 - eps)
        neg = torch.clamp(torch.sigmoid(neg), eps, 1.0 - eps)
        loss = torch.mean(torch.log(pos)) + torch.mean(torch.log(1 - neg))
        return -loss

    def _mine_loss(self, pos, neg):
        loss = torch.mean(pos) - torch.log(torch.mean(torch.exp(neg)))
        return -loss

    def _nce_loss(self, pos, neg):
        loss = torch.log(torch.exp(pos) + torch.sum(torch.exp(neg)))
        loss = torch.mean(pos - loss)
        return -loss

    def compute_loss(self, Y):
        C1, C2, CR = self._extract_chunks(Y)

        pos = self.discriminator(torch.cat((C1, C2), dim=1))
        neg = self.discriminator(torch.cat((C1, CR), dim=1))

        loss = 0
        if self.loss_fn == 'bce': loss = self._bce_loss(pos, neg)
        elif self.loss_fn == 'mine': loss = self._mine_loss(pos, neg)
        elif self.loss_fn == 'nce': loss = self._nce_loss(pos, neg)

        accuracy = torch.sum(pos > neg) / Y.size(0)
            
        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics
