from torch import nn
from torch import Tensor as T
import torch.nn.functional as F


class BYOLLoss(nn.Module):
    """
    BYOL loss.
    """

    def __init__(self):
        """
        Initialize a BYOL loss.

        Returns:
            None
        """
        super().__init__()

    def forward(self, P: T, Z: T) -> T:
        """
        Compute loss.

        Args:
            P (T): Embeddings tensor of predictor.
            Z (T): Embeddings tensor of projector.

        Returns:
            T: Loss tensor.
        """
        return 2 - 2 * F.cosine_similarity(P, Z.detach(), dim=-1).mean()
