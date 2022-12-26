import torch
from torch import nn
import torch.nn.functional as F


class BarlowTwinsLoss(nn.Module):

    def __init__(self, lamda=0.05, scale=0.025):
        super().__init__()

        self.lamda = lamda
        self.scale = scale

    def forward(self, Z_a, Z_b):
        N, D = Z_a.size()

        bn = nn.BatchNorm1d(D, affine=False).to(Z_a.device)
        Z_a = bn(Z_a)
        Z_b = bn(Z_b)
        
        c = (Z_a.T @ Z_b) / N

        diag = torch.eye(D, device=Z_a.device)

        loss = (c - diag).pow(2)
        loss[~diag.bool()] *= self.lamda
        return loss.sum() * self.scale
