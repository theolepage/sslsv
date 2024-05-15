from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T


class MoCoLoss(nn.Module):
    """
    MoCo loss.

    Attributes:
        temperature (float): Temperature value.
    """

    def __init__(self, temperature: float = 0.2):
        """
        Initialize a MoCo loss.

        Args:
            temperature (float): Temperature value. Defaults to 0.2.

        Returns:
            None
        """
        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        query: T,
        key: T,
        queue: T,
        current_labels: Optional[T] = None,
        queue_labels: Optional[T] = None,
    ) -> T:
        """
        Compute loss.

        Args:
            query (T): Query tensor.
            key (T): Key tensor.
            queue (T): Queue tensor.
            current_labels (Optional[T]): Labels tensor from the query/key.
            queue_labels (Optional[T]): Labels tensor from the queue.

        Returns:
            T: Loss tensor.
        """
        N, _ = query.size()

        pos = torch.einsum("nc,nc->n", (query, key)).unsqueeze(-1)
        neg = torch.einsum("nc,ck->nk", (query, queue))

        # Prevent class collisions using labels
        if current_labels is not None and queue_labels is not None:
            mask = current_labels.unsqueeze(1) == queue_labels.unsqueeze(0)
            neg[mask] = 0

        logits = torch.cat((pos, neg), dim=1) / self.temperature

        labels = torch.zeros(N, device=query.device, dtype=torch.long)

        return F.cross_entropy(logits, labels)
