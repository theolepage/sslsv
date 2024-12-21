from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods.SimCLRMargins.SimCLRMargins import SimCLRMargins, SimCLRMarginsConfig

from .SimCLRMultiViewsLoss import SimCLRMultiViewsLoss


@dataclass
class SimCLRMultiViewsConfig(SimCLRMarginsConfig):
    """
    SimCLR MultiViews method configuration.
    """

    pass


class SimCLRMultiViews(SimCLRMargins):
    """
    SimCLR MultiViews method.

    Attributes:
        loss_fn (SimCLRMultiViewsLoss): Loss function.
    """

    def __init__(
        self,
        config: SimCLRMultiViewsConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a SimCLR MultiViews method.

        Args:
            config (SimCLRMultiViewsConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.loss_fn = SimCLRMultiViewsLoss(config.loss)

    def forward(self, X: T, training: bool = False) -> T:
        """
        Forward pass.

        Args:
            X (T): Input tensor.
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            T: Encoder output for inference or embeddings for training.
        """
        if not training:
            return self.encoder(X)

        N, V, L = X.shape
        X = X.transpose(0, 1)
        global_frames = X[:2, :, :].reshape(-1, L)
        local_frames = X[2:, :, : L // 2].reshape(-1, L // 2)

        Y_global = self.encoder(global_frames)
        Y_local = self.encoder(local_frames)

        Z_global = (
            self.projector(Y_global) if self.config.enable_projector else Y_global
        )
        Z_local = self.projector(Y_local) if self.config.enable_projector else Y_local

        D = Z_global.size(-1)

        Z_global = Z_global.reshape(-1, N, D)
        Z_local = Z_local.reshape(-1, N, D)

        Z = torch.cat((Z_global, Z_local), dim=0).transpose(0, 1)

        return Z

    def train_step(
        self,
        Z: T,
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (T): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        margin = self.loss_fn.update_margin(self.epoch, self.max_epochs)

        loss = self.loss_fn(Z)

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/margin": margin,
            },
        )

        return loss
