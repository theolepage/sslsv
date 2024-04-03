import torch
from torch import nn
import torch.nn.functional as F


class DeepClusterLoss(nn.Module):

    def __init__(self, temperature=0.1):
        super().__init__()

        self.temperature = temperature

    def forward(self, preds, assignments):
        P, V, N, C = preds.size()

        loss = 0
        for p in range(P):
            logits = preds[p].view(-1, C) / self.temperature # (V*N, C)

            targets = assignments[p].repeat(V) # (V*N)
            targets = targets.to(preds.device, non_blocking=True)
            
            loss += F.cross_entropy(logits, targets, ignore_index=-1)

        return loss / P