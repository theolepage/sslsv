from torch import Tensor as T

from sslsv.methods.SimCLRMargins.SimCLRMarginsLoss import (
    SimCLRMarginsLoss,
    SimCLRMarginsLossConfig,
)


class SimCLRMultiViewsLoss(SimCLRMarginsLoss):
    """
    SimCLR MultiViews loss.

    Attributes:
        config (SimCLRMarginsLossConfig): Loss configuration.
        loss_fn (LossFunction): Loss function.
    """

    def __init__(self, config: SimCLRMarginsLossConfig):
        """
        Initialize a SimCLR MultiViews loss.

        Args:
            config (SimCLRMarginsLossConfig): Loss configuration.

        Returns:
            None
        """
        super().__init__(config)

    def forward(self, Z: T) -> T:
        """
        Compute loss.

        Args:
            Z (T): Embeddings tensor. Shape: (N, V, D).

        Returns:
            T: Loss tensor.
        """
        global_embeddings = Z[:, :2]
        local_embeddings = Z[:, 2:]

        loss = self.loss_fn(local_embeddings, global_embeddings, discard_identity=False)

        return loss
