from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from sslsv.utils.distributed import gather

from .WMSELoss import WMSELoss
from .Whitening2d import Whitening2d


@dataclass
class WMSEConfig(BaseSiameseMethodConfig):
    """
    W-MSE method configuration.

    Attributes:
        projector_hidden_dim (int): Projector hidden dimension.
        projector_output_dim (int): Projector output dimension.
        whitening_iters (int): Number of iterations for whitening.
        whitening_size (int): Size of the whitening matrix.
        whitening_eps (float): Epsilon value for numerical stability.
    """

    projector_hidden_dim: int = 1024
    projector_output_dim: int = 64

    whitening_iters: int = 1
    whitening_size: int = 128
    whitening_eps: float = 0.0


class WMSE(BaseSiameseMethod):
    """
    W-MSE (Whitening MSE) method.

    Paper:
        Whitening for Self-Supervised Representation Learning
        *Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, Nicu Sebe*
        ICML 2021
        https://arxiv.org/abs/2007.06346

    Attributes:
        projector (nn.Sequential): Projector module.
        whitening (Whitening2d): Whitening module.
        loss_fn (WMSELoss): Loss function.
    """

    def __init__(
        self,
        config: WMSEConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a W-MSE method.

        Args:
            config (WMSEConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
        )

        self.whitening = Whitening2d(
            config.projector_output_dim,
            eps=config.whitening_eps,
        )

        self.loss_fn = WMSELoss()

    def train_step(
        self,
        Z: Tuple[T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (Tuple[T, T]): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        Z_A = gather(Z[0])
        Z_B = gather(Z[1])

        N = Z_A.size(0)

        Z = torch.cat((Z_A, Z_B))

        loss = 0
        for _ in range(self.config.whitening_iters):
            z = torch.empty_like(Z)
            perm = torch.randperm(N).view(-1, self.config.whitening_size)
            for idx in perm:
                z[idx] = self.whitening(Z[idx])
                z[idx + N] = self.whitening(Z[idx + N])
            loss += self.loss_fn(z[:N], z[N : 2 * N])
        loss /= self.config.whitening_iters

        z1_std = F.normalize(Z_A, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(Z_B, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/z_std": z_std.detach(),
            },
        )

        return loss
