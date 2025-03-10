import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.utils.distributed import gather


class VICRegLoss(nn.Module):
    """
    VICReg loss.

    Attributes:
        inv_weight (float): Weight of the invariance loss term.
        var_weight (float): Weight of the variance loss term.
        cov_weight (float): Weight of the covariance loss term.
    """

    def __init__(
        self,
        inv_weight: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
    ):
        """
        Initialize a VICReg loss.

        Args:
            inv_weight (float): Weight of the invariance loss term. Defaults to 1.0.
            var_weight (float): Weight of the variance loss term. Defaults to 1.0.
            cov_weight (float): Weight of the covariance loss term. Defaults to 0.04.

        Returns:
            None
        """
        super().__init__()

        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, Z_a: T, Z_b: T) -> T:
        """
        Compute loss.

        Args:
            Z_a (T): Embeddings tensor of view A.
            Z_b (T): Embeddings tensor of view B.

        Returns:
            T: Loss tensor.
        """
        # Invariance loss
        inv_loss = F.mse_loss(Z_a, Z_b)

        Z_a = gather(Z_a)
        Z_b = gather(Z_b)

        # Variance loss
        Z_a_std = torch.sqrt(Z_a.var(dim=0) + 1e-04)
        Z_b_std = torch.sqrt(Z_b.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1 - Z_a_std))
        var_loss += torch.mean(F.relu(1 - Z_b_std))

        # Covariance loss
        N, D = Z_a.size()
        Z_a = Z_a - Z_a.mean(dim=0)
        Z_b = Z_b - Z_b.mean(dim=0)
        Z_a_cov = (Z_a.T @ Z_a) / (N - 1)
        Z_b_cov = (Z_b.T @ Z_b) / (N - 1)

        diag = torch.eye(D, dtype=torch.bool, device=Z_a.device)
        cov_loss = Z_a_cov[~diag].pow_(2).sum() / D
        cov_loss += Z_b_cov[~diag].pow_(2).sum() / D

        loss = self.inv_weight * inv_loss
        loss += self.var_weight * var_loss
        loss += self.cov_weight * cov_loss
        return loss
