import torch
from torch import nn
import torch.nn.functional as F

from sslsv.utils.distributed import gather, get_rank, get_world_size


class SimCLRLoss(nn.Module):

    def __init__(self, temperature=0.2):
        super().__init__()

        self.temperature = temperature
    
    def _create_masks(N):
        indexes = torch.cat((torch.arange(N), torch.arange(N)), dim=0)
        p1 = N * 2 * get_rank()
        p2 = N * 2 * (get_rank() + 1)

        # Create a mask with the same shape as the similarity matrix
        # and by considering all pairs as negatives by default
        pos_mask = torch.zeros((
            N * 2,
            N * 2 * get_world_size()
        ), dtype=torch.bool)
        
        # Define all pairs coming from the same sample as positives
        # for the current rank (GPU)
        pos_mask[:, p1:p2] = (indexes.unsqueeze(0) == indexes.unsqueeze(1))

        neg_mask = ~pos_mask

        # Discard positives of the same view
        pos_mask[:, p1:p2].fill_diagonal_(False)

        return pos_mask, neg_mask

    def forward(self, Z_1, Z_2):
        N, D = Z_1.size()

        Z = torch.cat((Z_1, Z_2), dim=0)
        Z = F.normalize(Z, p=2, dim=1)

        sim = (Z @ gather(Z).T) / self.temperature

        pos_mask, neg_mask = self._create_masks(N)

        pos = sim[pos_mask].view(N * 2, -1) # (N*2, 1) positives
        neg = sim[neg_mask].view(N * 2, -1) # (N*2, N*2*world_size-2) negatives

        logits = torch.cat((pos, neg), dim=1)
        labels = torch.zeros(N * 2, dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)