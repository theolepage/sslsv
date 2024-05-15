from torch import nn
import torch.nn.functional as F
from torch import Tensor as T


class DeepClusterLoss(nn.Module):
    """
    DeepCluster loss.

    Attributes:
        temperature (float): Temperature value.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize a DeepCluster loss.

        Args:
            temperature (float): Temperature value. Defaults to 0.1.

        Returns:
            None
        """
        super().__init__()

        self.temperature = temperature

    def forward(self, preds: T, assignments: T) -> T:
        """
        Compute loss.

        Args:
            preds (T): Predictions tensor. Shape: (P, V, N, D).
            assignments (T): Assignment tensor. Shape: (P, V, N).

        Returns:
            T: Loss tensor.
        """
        P, V, N, C = preds.size()

        loss = 0
        for p in range(P):
            logits = preds[p].view(-1, C) / self.temperature  # (V*N, C)

            targets = assignments[p].repeat(V)  # (V*N)
            targets = targets.to(preds.device, non_blocking=True)

            loss += F.cross_entropy(logits, targets, ignore_index=-1)

        return loss / P
