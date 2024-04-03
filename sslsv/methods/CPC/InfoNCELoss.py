import torch
from torch import nn
import torch.nn.functional as F

from sslsv.utils.distributed import gather, get_rank, get_world_size


class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.2):
        super().__init__()

        self.temperature = temperature

    @staticmethod
    def dot(Z_a, Z_b):
        return F.normalize(Z_a, p=2, dim=1) @ F.normalize(Z_b, p=2, dim=1).T

    @staticmethod
    def determine_accuracy(Z_a, Z_b):
        N, D = Z_a.size()

        dot = InfoNCELoss.dot(Z_a, Z_b)
        labels = torch.arange(N, device=dot.device)

        pred_indices = torch.argmax(F.softmax(dot, dim=1), dim=1)
        preds_acc = torch.eq(pred_indices, labels)
        accuracy = torch.count_nonzero(preds_acc) / N
        return accuracy

    def _create_masks(self, N):
        indexes = torch.arange(N)
        p1 = N * get_rank()
        p2 = N * (get_rank() + 1)

        pos_mask = torch.zeros((
            N ,
            N * get_world_size()
        ), dtype=torch.bool)
        
        pos_mask[:, p1:p2] = (indexes.unsqueeze(0) == indexes.unsqueeze(1))

        neg_mask = ~pos_mask

        return pos_mask, neg_mask

    def forward(self, Z_a, Z_b):
        N, D = Z_a.size()

        dot = InfoNCELoss.dot(Z_a, gather(Z_b)) / self.temperature

        pos_mask, neg_mask = self._create_masks(N)

        pos = dot[pos_mask].view(N, -1) # (N, 1) positives
        neg = dot[neg_mask].view(N, -1) # (N, N*world_size-1) negatives

        logits = torch.cat((pos, neg), dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=dot.device)

        return F.cross_entropy(logits, labels)