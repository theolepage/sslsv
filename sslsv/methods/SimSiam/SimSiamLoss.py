from torch import nn
import torch.nn.functional as F
from torch import Tensor as T


class SimSiamLoss(nn.Module):
    """
    SimSiam loss.
    """

    def __init__(self):
        """
        Initialize a SimSiam loss.

        Returns:
            None
        """
        super().__init__()

    def forward(self, P: T, Z: T) -> T:
        """
        Compute loss.

        Args:
            P (T): Predictions tensor.
            Z (T): Embeddings tensor.

        Returns:
            T: Loss tensor.
        """
        return -F.cosine_similarity(P, Z.detach(), dim=-1).mean()
