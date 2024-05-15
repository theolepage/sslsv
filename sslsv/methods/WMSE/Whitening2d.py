import torch
import torch.nn as nn
import torch.nn.functional as F


class Whitening2d(nn.Module):
    """
    Computes hard whitening for W-MSE in 2D using the Cholesky decomposition.

    Adapted from https://github.com/htdt/self-supervised/blob/master/methods/whitening.py.

    Attributes:
        output_dim (int): Number of dimension of projected features.
        eps (float): Epsilon value for numerical stability.
    """

    def __init__(self, output_dim: int, eps: float = 0.0):
        """
        Initialize a Whitening 2D object.

        Args:
            output_dim (int): Number of dimension of projected features.
            eps (float): Epsilon value for numerical stability. Defaults to 0.0.

        Returns:
            None
        """

        super().__init__()

        self.output_dim = output_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute whitening.

        Args:
            x (torch.Tensor): Input tensor. Shape: (N, D, H, W).

        Returns:
            torch.Tensor: Output tensor. Shape: (N, D, H, W).
        """
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.output_dim, -1).mean(-1).view(1, -1, 1, 1)
        xn = x - m

        T = xn.permute(1, 0, 2, 3).contiguous().view(self.output_dim, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)

        eye = torch.eye(self.output_dim).type(f_cov.type())

        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.linalg.solve_triangular(
            torch.linalg.cholesky(f_cov_shrinked), eye, upper=False
        )

        inv_sqrt = inv_sqrt.contiguous().view(self.output_dim, self.output_dim, 1, 1)

        decorrelated = F.conv2d(xn, inv_sqrt)

        return decorrelated.squeeze(2).squeeze(2)
