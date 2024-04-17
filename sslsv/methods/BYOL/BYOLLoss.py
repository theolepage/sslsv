from torch import nn
from torch import Tensor as T
import torch.nn.functional as F


class BYOLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, P: T, Z: T) -> T:
        return 2 - 2 * F.cosine_similarity(P, Z.detach(), dim=-1).mean()
