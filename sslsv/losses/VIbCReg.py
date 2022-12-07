import torch
from torch import nn
import torch.nn.functional as F


class VIbCRegLoss(nn.Module):

    def __init__(
        self,
        inv_weight=1.0,
        var_weight=1.0,
        cov_weight=8.0
    ):
        super().__init__()

        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, data):
        Z_a, Z_b = data

        N, D = Z_a.size()

        # Invariance loss
        inv_loss = F.mse_loss(Z_a, Z_b)

        # Variance loss
        Z_a_std = torch.sqrt(Z_a.var(dim=0) + 1e-04)
        Z_b_std = torch.sqrt(Z_b.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1 - Z_a_std))
        var_loss += torch.mean(F.relu(1 - Z_b_std))

        # Covariance loss
        Z_a = Z_a - Z_a.mean(dim=0)
        Z_b = Z_b - Z_b.mean(dim=0)
        Z_a_norm = F.normalize(Z_a, p=2, dim=0)
        Z_b_norm = F.normalize(Z_b, p=2, dim=0)
        Z_a_cov = Z_a_norm.T @ Z_a_norm
        Z_b_cov = Z_b_norm.T @ Z_b_norm

        Z_a_cov.fill_diagonal_(0.0)
        Z_b_cov.fill_diagonal_(0.0)
        
        cov_loss = (Z_a_cov ** 2).mean()
        cov_loss += (Z_b_cov ** 2).mean()

        loss = self.inv_weight * inv_loss
        loss += self.var_weight * var_loss
        loss += self.cov_weight * cov_loss
        return loss