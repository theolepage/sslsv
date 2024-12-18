from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import math

import torch
from torch import nn
import torch.nn.functional as F
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
        loss_symmetric (bool): Whether to use symmetric formulation of NT-Xent.
        scale (float): Scale factor for the loss function.
        margin (float): Margin value for the loss function.
        margin_learnable (bool): Whether the margin value is learnable.
        margin_scheduler (bool): Whether to use a scheduler for the margin value.
        margin_simo (bool): Whether to use SIMO as the margin.
        margin_simo_K (int): K value for the SIMO margin.
        margin_simo_alpha (int): Alpha value for the SIMO margin.
        magface_l_margin (float): MagFace lower margin.
        magface_u_margin (float): MagFace lpper margin.
        magface_l_a (int): MagFace lower norm.
        magface_u_a (int): MagFace upper norm.
        magface_lambda_g (float): MagFace weight for regularization.
        adaface_h (float): AdaFace hyper-parameter h.
        loss_reg_weight (float): Weight of the MHE regularization term in the loss function.
        enable_projector (bool): Whether to enable use a projector.
        projector_hidden_dim (int): Hidden dimension of the projector.
        projector_output_dim (int): Output dimension of the projector.
    """

    enable_multi_views: bool = False

    loss: SimCLRMarginsLossEnum = SimCLRMarginsLossEnum.NTXENT
    loss_symmetric: bool = True

    scale: float = 5

    margin: float = 0.2
    margin_learnable: bool = False
    margin_scheduler: bool = False

    # SIMO
    margin_simo: bool = False
    margin_simo_K: int = 2 * 255
    margin_simo_alpha: int = 65536

    # MagFace
    magface_l_margin: float = 0.01
    magface_u_margin: float = 0.05
    magface_l_a: int = 10
    magface_u_a: int = 110
    magface_lambda_g: float = 0

    # AdaFace
    adaface_h: float = 0.333

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

        self.embeddings_dim = self.encoder.encoder_dim

        if config.enable_projector:
            self.embeddings_dim = config.projector_output_dim
            self.projector = nn.Sequential(
                nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            )

        self.loss_fn = SimCLRMarginsLoss(
            enable_multi_views=config.enable_multi_views,
            loss=config.loss,
            symmetric=config.loss_symmetric,
            scale=config.scale,
            margin=config.margin,
            margin_simo=config.margin_simo,
            margin_simo_K=config.margin_simo_K,
            margin_simo_alpha=config.margin_simo_alpha,
            magface_l_margin=config.magface_l_margin,
            magface_u_margin=config.magface_u_margin,
            magface_l_a=config.magface_l_a,
            magface_u_a=config.magface_u_a,
            magface_lambda_g=config.magface_lambda_g,
            adaface_h=config.adaface_h,
            margin_learnable=config.margin_learnable,
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

        # if self.config.enable_multi_views:
        #     N, V, L = X.shape
        #     X = X.transpose(0, 1)
        #     global_frames = X[:2, :, :].reshape(-1, L)
        #     local_frames = X[2:, :, : L // 2].reshape(-1, L // 2)

        #     Z_global = self._compute_embeddings(global_frames)
        #     Z_local = self._compute_embeddings(local_frames)

        #     D = Z_global.size(-1)

        #     Z_global = Z_global.reshape(-1, N, D)
        #     Z_local = Z_local.reshape(-1, N, D)

        #     Z = torch.cat((Z_global, Z_local), dim=0).transpose(0, 1)
        # else:
        #   X_1 = X[:, 0, :]
        #   X_2 = X[:, 1, :]
        #   views = [X_1, X_2]
        #   Z = torch.stack([self._compute_embeddings(V) for V in views], dim=1)

        frame_length = self.trainer.config.dataset.frame_length
        X_1 = X[:, 0, :frame_length]
        X_2 = X[:, 1, :frame_length]

        Z_1 = self._compute_embeddings(X_1)
        Z_2 = self._compute_embeddings(X_2)

        Z_ssps = None
        if self.ssps:
            encoder_training_mode = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                Z_ssps = F.normalize(self.encoder(X[:, -1]).detach(), p=2, dim=-1)
            if encoder_training_mode:
                self.encoder.train()

        return Z_1, Z_2, Z_ssps

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = []
        if self.config.margin_learnable:
            extra_learnable_params += [{"params": self.loss_fn.parameters()}]
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
        super().on_train_epoch_start(epoch, max_epochs)

        self.epoch = epoch
        self.max_epochs = max_epochs

    def _margin_scheduler(self):
        """
        Loss margin cosine scheduler based on epoch.

        Returns:
            float: Margin value.
        """
        if self.epoch > (self.max_epochs // 2):
            return self.config.margin

        return (
            self.config.margin
            - self.config.margin
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
        Z_1, Z_2, Z_ssps = Z

        margin = self.config.margin
        if self.config.margin_scheduler:
            margin = self._margin_scheduler()
            self.loss_fn.loss_fn.margin = margin

        if self.ssps:
            self.ssps.sample(indices=indices, embeddings=Z_ssps)
            Z_2_pp = self.ssps.apply(0, Z_2)
            self.ssps.update_buffers(step_rel, indices, Z_ssps, [Z_2])
            loss = self.loss_fn(
                Z_1,
                Z_2_pp,
                # ssps_assignments=self.ssps.sampling.assignments[indices],
            )
        else:
            loss = self.loss_fn(Z_1, Z_2)

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/margin": margin,
            },
        )

        return loss
