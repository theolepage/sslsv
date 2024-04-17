from enum import Enum

import torch
from torch import nn
from torch import Tensor as T


class LIMLossEnum(Enum):

    BCE = "bce"
    NCE = "nce"
    MINE = "mine"


class LIMLoss(nn.Module):

    def __init__(self, loss: LIMLossEnum):
        super().__init__()

        _LOSS_METHODS = {
            LIMLossEnum.BCE: LIMLoss._bce_loss,
            LIMLossEnum.MINE: LIMLoss._mine_loss,
            LIMLossEnum.NCE: LIMLoss._nce_loss,
        }

        self.loss_fn = _LOSS_METHODS[loss]

    @staticmethod
    def _bce_loss(pos: T, neg: T, eps: float = 1e-07) -> T:
        pos = torch.clamp(torch.sigmoid(pos), eps, 1.0 - eps)
        neg = torch.clamp(torch.sigmoid(neg), eps, 1.0 - eps)
        loss = torch.mean(torch.log(pos)) + torch.mean(torch.log(1 - neg))
        return -loss

    @staticmethod
    def _mine_loss(pos: T, neg: T) -> T:
        loss = torch.mean(pos) - torch.log(torch.mean(torch.exp(neg)))
        return -loss

    @staticmethod
    def _nce_loss(pos: T, neg: T) -> T:
        loss = torch.log(torch.exp(pos) + torch.sum(torch.exp(neg)))
        loss = torch.mean(pos - loss)
        return -loss

    def forward(self, pos: T, neg: T) -> T:
        return self.loss_fn(pos, neg)
