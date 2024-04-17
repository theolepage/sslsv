from torch import nn
import torch.nn.functional as F
from torch import Tensor as T


class WMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Z_a: T, Z_b: T) -> T:
        return 2 - 2 * (F.normalize(Z_a) * F.normalize(Z_b)).sum(dim=-1).mean()
