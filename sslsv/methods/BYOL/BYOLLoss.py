import torch
from torch import nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, P, Z):
        return 2 - 2 * F.cosine_similarity(P, Z.detach(), dim=-1).mean()