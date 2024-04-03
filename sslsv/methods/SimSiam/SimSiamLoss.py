import torch
from torch import nn
import torch.nn.functional as F


class SimSiamLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, P, Z):
        return -F.cosine_similarity(P, Z.detach(), dim=-1).mean()