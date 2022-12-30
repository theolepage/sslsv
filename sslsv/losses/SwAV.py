import torch
from torch import nn
import torch.nn.functional as F


class SwAVLoss(nn.Module):

    def __init__(self, temperature=0.1):
        super().__init__()

        self.temperature = temperature

    def forward(self, preds, assignments):
        losses = []
        for i, A in enumerate(assignments):
            for j, P in enumerate(preds):
                if i == j: continue

                P = P / self.temperature
                loss = -torch.mean(torch.sum(A * F.log_softmax(P, dim=1), dim=1))
                losses.append(loss)

        return sum(losses) / len(losses)