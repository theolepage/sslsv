from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import math

import torch
from torch import nn
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .SimCLRMarginsLoss import SimCLRMarginsLoss, SimCLRMarginsLossEnum


@dataclass
class SimCLRMarginsConfig(BaseMethodConfig):
    """
    SimCLR Margins method configuration.

    Attributes:
        enable_multi_views (bool): Whether to enable multi views training.
        loss (SimCLRMarginsLossEnum): Type of loss function.
        loss_scale (float): Scale factor for the loss function.
        loss_margin (float): Margin value for the loss function.
        loss_margin_simo (bool): Whether to use SIMO as the margin.
        loss_margin_simo_K (int): K value for the SIMO margin.
        loss_margin_simo_alpha (int): Alpha value for the SIMO margin.
        loss_margin_learnable (bool): Whether the margin value is learnable.
        loss_margin_scheduler (bool): Whether to use a scheduler for the margin value.
        loss_reg_weight (float): Weight of the MHE regularization term in the loss function.
        enable_projector (bool): Whether to enable use a projector.
        projector_hidden_dim (int): Hidden dimension of the projector.
        projector_output_dim (int): Output dimension of the projector.
    """

    enable_multi_views: bool = False

    loss: SimCLRMarginsLossEnum = SimCLRMarginsLossEnum.SNTXENT

    loss_scale: float = 5

    loss_margin: float = 0.2

    loss_margin_simo: bool = False
    loss_margin_simo_K: int = 2 * 255
    loss_margin_simo_alpha: int = 65536

    loss_margin_learnable: bool = False

    loss_margin_scheduler: bool = False

    loss_reg_weight: float = 0.0

    enable_projector: bool = True
    projector_hidden_dim: int = 2048
    projector_output_dim: int = 256


class SimCLRMargins(BaseMethod):
    """
    SimCLR Margins method.

    Papers:
        - Experimenting with Additive Margins for Contrastive Self-Supervised Speaker Verification
          *Théo Lepage, Réda Dehak*
          https://arxiv.org/abs/2306.03664

        - Additive Margin in Contrastive Self-Supervised Frameworks to Learn Discriminative Speaker Representations
          *Théo Lepage, Réda Dehak*
          https://arxiv.org/abs/2404.14913

    Attributes:
        epoch (int): Current epoch.
        max_epochs (int): Maximum number of epochs.
        projector (nn.Sequential): Projector module.
        loss_fn (SimCLRMarginsLoss): Loss function.
    """

    def __init__(
        self,
        config: SimCLRMarginsConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a SimCLR Margins method.

        Args:
            config (SimCLRMarginsConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.epoch = 0
        self.max_epochs = 0

        if config.enable_projector:
            self.projector = nn.Sequential(
                nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            )

        self.loss_fn = SimCLRMarginsLoss(
            enable_multi_views=config.enable_multi_views,
            loss=config.loss,
            loss_scale=config.loss_scale,
            loss_margin=config.loss_margin,
            loss_margin_simo=config.loss_margin_simo,
            loss_margin_simo_K=config.loss_margin_simo_K,
            loss_margin_simo_alpha=config.loss_margin_simo_alpha,
            loss_margin_learnable=config.loss_margin_learnable,
            loss_reg_weight=config.loss_reg_weight,
        )

    def _compute_embeddings(self, X: T) -> T:
        """
        Compute embeddings for training.

        Args:
            X (T): Input tensor.

        Returns:
            T: Output tensor.
        """
        Y = self.encoder(X)

        if self.config.enable_projector:
            return self.projector(Y)

        return Y

    def forward(self, X: T, training: bool = False) -> T:
        """
        Forward pass.

        Args:
            X (T): Input tensor
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            T: Encoder output for inference or embeddings for training.
        """
        if not training:
            return self.encoder(X)

        if self.config.enable_multi_views:
            N, V, L = X.shape
            X = X.transpose(0, 1)
            global_frames = X[:2, :, :].reshape(-1, L)
            local_frames = X[2:, :, : L // 2].reshape(-1, L // 2)

            Z_global = self._compute_embeddings(global_frames)
            Z_local = self._compute_embeddings(local_frames)

            D = Z_global.size(-1)

            Z_global = Z_global.reshape(-1, N, D)
            Z_local = Z_local.reshape(-1, N, D)

            Z = torch.cat((Z_global, Z_local), dim=0).transpose(0, 1)
        else:
            X_1 = X[:, 0, :]
            X_2 = X[:, 1, :]
            views = [X_1, X_2]
            Z = torch.stack([self._compute_embeddings(V) for V in views], dim=1)

        return Z

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = [{"params": self.loss_fn.parameters()}]
        if self.config.enable_projector:
            extra_learnable_params += [
                {"params": self.projector.parameters()},
            ]
        return super().get_learnable_params() + extra_learnable_params

    def on_train_epoch_start(self, epoch: int, max_epochs: int):
        """
        Update epoch and max_epochs values for margin scheduler.

        Args:
            epoch (int): Current epoch.
            max_epochs (int): Total number of epochs.

        Returns:
            None
        """
        self.epoch = epoch
        self.max_epochs = max_epochs

    def _loss_margin_scheduler(self):
        """
        Loss margin cosine scheduler based on epoch.

        Returns:
            float: Margin value.
        """
        if self.epoch > (self.max_epochs // 2):
            return self.config.loss_margin

        return (
            self.config.loss_margin
            - self.config.loss_margin
            * (math.cos(math.pi * self.epoch / (self.max_epochs // 2)) + 1)
            / 2
        )

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
        loss_margin = self.config.loss_margin

        if self.config.loss_margin_scheduler:
            loss_margin = self._loss_margin_scheduler()
            self.loss_fn.loss_fn.margin = loss_margin

        loss = self.loss_fn(Z)

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/margin": loss_margin,
            },
        )

        return loss
