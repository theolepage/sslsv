from enum import Enum

import torch
from torch import nn
from torch import Tensor as T


class LIMLossEnum(Enum):
    """
    Enumeration representing different types of Loss functions for LIM.

    Attributes:
        BCE (str): Binary Cross-Entropy (BCE) loss.
        NCE (str): Noise Contrastive Estimation (NCE) loss.
        MINE (str): Mutual Information Neural Estimation (MINE) loss.
    """

    BCE = "bce"
    NCE = "nce"
    MINE = "mine"


class LIMLoss(nn.Module):
    """
    LIM loss.

    Attributes:
        loss_fn (Callable[..., T]): Loss function.
    """

    def __init__(self, loss: LIMLossEnum):
        """
        Initialize a LIM loss.

        Args:
            loss (LIMLossEnum): Loss function option.

        Returns:
            None
        """
        super().__init__()

        _LOSS_METHODS = {
            LIMLossEnum.BCE: LIMLoss._bce_loss,
            LIMLossEnum.MINE: LIMLoss._mine_loss,
            LIMLossEnum.NCE: LIMLoss._nce_loss,
        }

        self.loss_fn = _LOSS_METHODS[loss]

    @staticmethod
    def _bce_loss(pos: T, neg: T, eps: float = 1e-07) -> T:
        """
        Compute the Binary Cross-Entropy (BCE) loss.

        Args:
            pos (T): Positive embeddings tensor.
            neg (T): Negative embeddings tensor.
            eps (float): Epsilon value for numerical stability. Defaults to 1e-07.

        Returns:
            T: Loss tensor.
        """
        pos = torch.clamp(torch.sigmoid(pos), eps, 1.0 - eps)
        neg = torch.clamp(torch.sigmoid(neg), eps, 1.0 - eps)
        loss = torch.mean(torch.log(pos)) + torch.mean(torch.log(1 - neg))
        return -loss

    @staticmethod
    def _mine_loss(pos: T, neg: T) -> T:
        """
        Compute the Mutual Information Neural Estimation (MINE) loss.

        Args:
            pos (T): Positive embeddings tensor.
            neg (T): Negative embeddings tensor.

        Returns:
            T: Loss tensor.
        """
        loss = torch.mean(pos) - torch.log(torch.mean(torch.exp(neg)))
        return -loss

    @staticmethod
    def _nce_loss(pos: T, neg: T) -> T:
        """
        Compute the Noise Contrastive Estimation (NCE) loss.

        Args:
            pos (T): Positive embeddings tensor.
            neg (T): Negative embeddings tensor.

        Returns:
            T: Loss tensor.
        """
        loss = torch.log(torch.exp(pos) + torch.sum(torch.exp(neg)))
        loss = torch.mean(pos - loss)
        return -loss

    def forward(self, pos: T, neg: T) -> T:
        """
        Compute loss.

        Args:
            pos (T): Positive embeddings tensor.
            neg (T): Negative embeddings tensor..

        Returns:
            T: Loss tensor.
        """
        return self.loss_fn(pos, neg)
