from torch import nn
from torch import Tensor as T

from .InfoNCELoss import InfoNCELoss


class CPCLoss(nn.Module):
    """
    CPC loss.

    Attributes:
        infonce (InfoNCELoss): InfoNCE loss.
    """

    def __init__(self):
        """
        Initialize a CPC loss.

        Returns:
            None
        """
        super().__init__()

        self.infonce = InfoNCELoss(temperature=1.0, normalize=False)

    def forward(self, Y_future_preds: T, Y_future: T) -> T:
        """
        Compute loss.

        Args:
            Y_future_preds (T): Predicted embeddings tensor.
            Y_future (T): Embeddings tensor.

        Returns:
            T: Loss tensor.
        """
        # Shape: (N, encoded_dim, nb_t_to_predict)

        nb_t_to_predict = Y_future.size(2)

        loss = 0
        for t in range(nb_t_to_predict):
            loss += self.infonce(
                Y_future_preds[:, :, t].contiguous(),
                Y_future[:, :, t].contiguous(),
            )

        loss /= nb_t_to_predict

        return loss
