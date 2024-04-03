import torch
from torch import nn
import torch.nn.functional as F


class MoCoLoss(nn.Module):

    def __init__(self, temperature=0.2):
        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        query,
        key,
        queue,
        current_labels=None,
        queue_labels=None
    ):
        N, _ = query.size()

        pos = torch.einsum('nc,nc->n', (query, key)).unsqueeze(-1)
        neg = torch.einsum("nc,ck->nk", (query, queue))

        # Prevent class collisions using labels
        if current_labels is not None and queue_labels is not None:
            mask = current_labels.unsqueeze(1) == queue_labels.unsqueeze(0)
            neg[mask] = 0

        logits = torch.cat((pos, neg), dim=1) / self.temperature

        labels = torch.zeros(N, device=query.device, dtype=torch.long)

        return F.cross_entropy(logits, labels)