import torch
from torch import nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):

    def __init__(self, temperature=0.2):
        super().__init__()

        self.temperature = temperature

    def forward(self, Z_1, Z_2):
        N, D = Z_1.size()

        feats = torch.cat((Z_1, Z_2), dim=0)
        feats = F.normalize(feats, p=2, dim=1)

        sim = feats @ feats.T

        labels = torch.cat((torch.arange(N), torch.arange(N)), dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Discard diagonal (i = j)
        mask = torch.eye(2 * N, dtype=torch.bool)
        labels = labels[~mask].view(2 * N, -1)
        sim = sim[~mask].view(2 * N, -1)

        pos = sim[labels.bool()].view(2 * N, -1)
        neg = sim[~labels.bool()].view(2 * N, -1)

        logits = torch.cat((pos, neg), dim=1) / self.temperature
        labels = torch.zeros(2 * N, dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)