import torch
from torch import nn
import torch.nn.functional as F


class WMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Z_a, Z_b):
        return 2 - 2 * (F.normalize(Z_a) * F.normalize(Z_b)).sum(dim=-1).mean()