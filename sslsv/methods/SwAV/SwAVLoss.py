import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T


class SwAVLoss(nn.Module):
    """
    SwAV loss.

    Attributes:
        temperature (float): Temperature value.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Initialize a SwAV loss.

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
            preds (T): Predictions tensor.
            assignments (T): Assignments tensor.

        Returns:
            T: Loss tensor.
        """
        losses = []
        for i, A in enumerate(assignments):
            for j, P in enumerate(preds):
                if i == j:
                    continue

                P = P / self.temperature
                loss = -torch.mean(torch.sum(A * F.log_softmax(P, dim=1), dim=1))
                losses.append(loss)

        return sum(losses) / len(losses)
