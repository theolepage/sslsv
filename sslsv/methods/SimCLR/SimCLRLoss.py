from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.utils.distributed import gather, get_rank, get_world_size


class SimCLRLoss(nn.Module):
    """
    SimCLR loss.

    Attributes:
        temperature (float): Temperature value.
    """

    def __init__(self, temperature: float = 0.2):
        """
        Initialize a SimCLR loss.

        Args:
            temperature (float): Temperature value. Defaults to 0.2.

        Returns:
            None
        """
        super().__init__()

        self.temperature = temperature

    @staticmethod
    def create_contrastive_masks(
        N: int,
        V_A: int,
        V_B: int,
        rank: int,
        world_size: int,
        discard_identity: bool = True,
    ) -> Tuple[T, T]:
        """
        Create masks to extract positives and negatives from the dot product result.

        Args:
            N (int): Size of the masks.
            V_A (int): Number of views in first embeddings tensor.
            V_B (int): Number of views in second embeddings tensor.
            rank (int): Rank of the current process (distributed).
            world_size (int): Total number of processes (distributed).
            discard_identity (bool): Whether to discard the identity. Defaults to True.

        Returns:
            Tuple[T, T]: Positive and negative masks.
        """
        # Create a mask with the same shape as the similarity matrix
        # and by considering all pairs as negatives by default
        pos_mask = torch.zeros((N * V_A, N * V_B * world_size), dtype=torch.bool)

        # Define all pairs coming from the same sample as positives
        # for the current rank (GPU)
        pos_mask_local = torch.arange(N).unsqueeze(0) == torch.arange(N).unsqueeze(1)
        pos_mask_local = pos_mask_local.repeat(V_A, V_B)
        # pos_mask_local: (V_A*N, V_B*N)
        # For each sample (row)
        #   - a column with 1 is a positive
        #   - a column with 0 is a negative
        p1 = N * V_B * rank
        p2 = N * V_B * (rank + 1)
        pos_mask[:, p1:p2] = pos_mask_local

        neg_mask = ~pos_mask

        # Discard positives of the same view
        if discard_identity:
            pos_mask[:, p1:p2].fill_diagonal_(False)

        return pos_mask, neg_mask

    def forward(self, Z_1: T, Z_2: T) -> T:
        """
        Compute loss.

        Args:
            Z_1 (T): Embeddings tensor of view 1.
            Z_2 (T): Embeddings tensor of view 2.

        Returns:
            T: Loss tensor.
        """
        N, D = Z_1.size()

        Z = torch.cat((Z_1, Z_2), dim=0)
        Z = F.normalize(Z, p=2, dim=1)

        sim = (Z @ gather(Z).T) / self.temperature
        # sim: (V_A*N, V_B*N)

        pos_mask, neg_mask = self.create_contrastive_masks(
            N=N,
            V_A=2,
            V_B=2,
            rank=get_rank(),
            world_size=get_world_size(),
        )

        pos = sim[pos_mask].view(N * 2, -1)  # (N*2, 1) positives
        neg = sim[neg_mask].view(N * 2, -1)  # (N*2, N*2*world_size-2) negatives

        logits = torch.cat((pos, neg), dim=1)
        labels = torch.zeros(N * 2, dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)
