import torch
from torch import nn
from torch import Tensor as T

import torch.distributed as dist
from sslsv.utils.distributed import is_dist_initialized, get_world_size


class BarlowTwinsLoss(nn.Module):

    def __init__(self, lamda: float = 0.05, scale: float = 0.025):
        super().__init__()

        self.lamda = lamda
        self.scale = scale

    def forward(self, Z_a: T, Z_b: T) -> T:
        N, D = Z_a.size()

        bn = nn.BatchNorm1d(D, affine=False).to(Z_a.device)
        Z_a = bn(Z_a)
        Z_b = bn(Z_b)

        c = (Z_a.T @ Z_b) / N

        if is_dist_initialized():
            dist.all_reduce(c)
            c /= get_world_size()

        diag = torch.eye(D, device=Z_a.device)

        loss = (c - diag).pow(2)
        loss[~diag.bool()] *= self.lamda
        return loss.sum() * self.scale
