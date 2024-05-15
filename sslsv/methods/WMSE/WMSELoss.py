from torch import nn
import torch.nn.functional as F
from torch import Tensor as T


class WMSELoss(nn.Module):
    """
    W-MSE loss.
    """

    def __init__(self):
        """
        Initialize a W-MSE loss.

        Returns:
            None
        """
        super().__init__()

    def forward(self, Z_a: T, Z_b: T) -> T:
        """
        Compute loss.

        Args:
            Z_a (T): Embeddings tensor of view A.
            Z_b (T): Embeddings tensor of view B.

        Returns:
            T: Loss tensor.
        """
        return 2 - 2 * (F.normalize(Z_a) * F.normalize(Z_b)).sum(dim=-1).mean()
