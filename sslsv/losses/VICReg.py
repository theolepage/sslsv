import torch
from torch import nn
import torch.nn.functional as F

from sslsv.utils.distributed import gather


class VICRegLoss(nn.Module):

    def __init__(
        self,
        inv_weight=1.0,
        var_weight=1.0,
        cov_weight=0.04
    ):
        super().__init__()

        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, Z_a, Z_b):
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