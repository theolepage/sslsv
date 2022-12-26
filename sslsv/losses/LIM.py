import torch
from torch import nn
import torch.nn.functional as F


class LIMLoss(nn.Module):

    def __init__(self, loss_name='bce'):
        super().__init__()

        _LOSS_METHODS = {
            'bce':  LIMLoss._bce_loss,
            'mine': LIMLoss._mine_loss,
            'nce':  LIMLoss._nce_loss
        }

        self.loss_fn = _LOSS_METHODS[loss_name]

    @staticmethod
    def _bce_loss(pos, neg, eps=1e-07):
        pos = torch.clamp(torch.sigmoid(pos), eps, 1.0 - eps)
        neg = torch.clamp(torch.sigmoid(neg), eps, 1.0 - eps)
        loss = torch.mean(torch.log(pos)) + torch.mean(torch.log(1 - neg))
        return -loss

    @staticmethod
    def _mine_loss(pos, neg):
        loss = torch.mean(pos) - torch.log(torch.mean(torch.exp(neg)))
        return -loss

    @staticmethod
    def _nce_loss(pos, neg):
        loss = torch.log(torch.exp(pos) + torch.sum(torch.exp(neg)))
        loss = torch.mean(pos - loss)
        return -loss

    def forward(self, pos, neg):
        return self.loss_fn(pos, neg)