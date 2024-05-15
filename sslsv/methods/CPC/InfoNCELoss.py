from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.utils.distributed import gather, get_rank, get_world_size


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss.
    """

    def __init__(self, temperature: float = 0.2, normalize: bool = True):
        """
        Initialize an InfoNCE loss.

        Args:
            temperature (float): Temperature value. Defaults to 0.2.
            normalize (bool): Whether to normalize the input tensors. Defaults to True.

        Returns:
            None
        """
        super().__init__()

        self.temperature = temperature
        self.normalize = normalize

    @staticmethod
    def dot(Z_a: T, Z_b: T, normalize: bool) -> T:
        """
        Compute dot product between two tensors.

        Args:
            Z_a (T): First input tensor.
            Z_b (T): Second input tensor.
            normalize (bool): If True, normalize the input tensors before computing the dot product.

        Returns:
            T: Output tensor.
        """
        if normalize:
            Z_a = F.normalize(Z_a, p=2, dim=1)
            Z_b = F.normalize(Z_b, p=2, dim=1)
        return Z_a @ Z_b.T

    @staticmethod
    def determine_accuracy(Z_a: T, Z_b: T) -> T:
        """
        Determine the accuracy of predictions.

        Args:
            Z_a (T): First input tensor.
            Z_b (T): Second input tensor.

        Returns:
            T: Accuracy tensor.
        """
        N, D = Z_a.size()

        dot = InfoNCELoss.dot(Z_a, Z_b, normalize=True)
        labels = torch.arange(N, device=dot.device)

        pred_indices = torch.argmax(F.softmax(dot, dim=1), dim=1)
        preds_acc = torch.eq(pred_indices, labels)
        accuracy = torch.count_nonzero(preds_acc) / N
        return accuracy

    def _create_masks(self, N: int) -> Tuple[T, T]:
        """
        Create masks to extract positives and negatives from the dot product result.

        Args:
            N (int): Size of the masks.

        Returns:
            Tuple[T, T]: Positive and negative masks.
        """
        indexes = torch.arange(N)
        p1 = N * get_rank()
        p2 = N * (get_rank() + 1)

        pos_mask = torch.zeros((N, N * get_world_size()), dtype=torch.bool)

        pos_mask[:, p1:p2] = indexes.unsqueeze(0) == indexes.unsqueeze(1)

        neg_mask = ~pos_mask

        return pos_mask, neg_mask

    def forward(self, Z_a: T, Z_b: T) -> T:
        """
        Compute loss.

        Args:
            Z_a (T): Embedding tensor of first view.
            Z_b (T): Embedding tensor of second view.

        Returns:
            T: Loss tensor.
        """
        N, D = Z_a.size()

        dot = InfoNCELoss.dot(Z_a, gather(Z_b), self.normalize) / self.temperature

        pos_mask, neg_mask = self._create_masks(N)

        pos = dot[pos_mask].view(N, -1)  # (N, 1) positives
        neg = dot[neg_mask].view(N, -1)  # (N, N*world_size-1) negatives

        logits = torch.cat((pos, neg), dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=dot.device)

        return F.cross_entropy(logits, labels)
