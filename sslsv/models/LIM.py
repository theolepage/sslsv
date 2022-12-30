import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
from enum import Enum

from sslsv.losses.LIM import LIMLoss
from sslsv.models._BaseModel import BaseModel, BaseModelConfig


class LIMLossFnEnum(Enum):

    BCE = 'bce'
    NCE = 'nce'
    MINE = 'mine'


@dataclass
class LIMConfig(BaseModelConfig):

    loss_name: LIMLossFnEnum = 'bce'
    context_length: int = 1


class LIM(BaseModel):

    def __init__(self, config, create_encoder_fn):
        super().__init__(config, create_encoder_fn)

        self.loss_name = config.loss_name
        self.context_length = config.context_length

        self.discriminator = nn.Sequential(
            nn.Linear(2 * self.encoder.encoder_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.loss_fn = LIMLoss(self.loss_name)

    def forward(self, X, training=False):
        if not training: return self.encoder(X).mean(dim=2)
        
        return self.encoder(X)

    def get_learnable_params(self):
        extra_learnable_params = [
            {'params': self.discriminator.parameters()}
        ]
        return super().get_learnable_params() + extra_learnable_params

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

    def train_step(self, Y, step, samples):
        C1, C2, CR = self._extract_chunks(Y)

        pos = self.discriminator(torch.cat((C1, C2), dim=1))
        neg = self.discriminator(torch.cat((C1, CR), dim=1))

        loss = self.loss_fn(pos, neg)

        accuracy = torch.sum(pos > neg) / Y.size(0)
            
        metrics = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }

        return loss, metrics
