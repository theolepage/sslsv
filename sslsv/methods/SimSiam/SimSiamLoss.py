from torch import nn
import torch.nn.functional as F
from torch import Tensor as T


class SimSiamLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, P: T, Z: T) -> T:
        return -F.cosine_similarity(P, Z.detach(), dim=-1).mean()
