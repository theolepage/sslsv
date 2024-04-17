from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import math

import torch
from torch import nn
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .SimCLRCustomLoss import SimCLRCustomLoss, SimCLRCustomLossEnum


@dataclass
class SimCLRCustomConfig(BaseMethodConfig):

    enable_multi_views: bool = False

    loss: SimCLRCustomLossEnum = SimCLRCustomLossEnum.SNTXENT

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


class SimCLRCustom(BaseMethod):

    def __init__(
        self,
        config: SimCLRCustomConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        super().__init__(config, create_encoder_fn)

        self.epoch = 0
        self.max_epochs = 0

        if config.enable_projector:
            self.projector = nn.Sequential(
                nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            )

        self.loss_fn = SimCLRCustomLoss(
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
        Y = self.encoder(X)

        if self.config.enable_projector:
            return self.projector(Y)

        return Y

    def forward(self, X: T, training: bool = False) -> Union[T, Tuple[T, ...]]:
        if not training:
            return self.encoder(X)

        # Retrieve global views
        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]
        views = [X_1, X_2]

        # Extract local views
        if self.config.enable_multi_views:
            small_frame_length = X.size(-1) // 2
            views.append(X_1[:, :small_frame_length])
            views.append(X_1[:, small_frame_length:])
            views.append(X_2[:, :small_frame_length])
            views.append(X_2[:, small_frame_length:])

        Z = torch.stack([self._compute_embeddings(V) for V in views], dim=1)

        return Z

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        extra_learnable_params = [{"params": self.loss_fn.parameters()}]
        if self.config.enable_projector:
            extra_learnable_params += [
                {"params": self.projector.parameters()},
            ]
        return super().get_learnable_params() + extra_learnable_params

    def on_train_epoch_start(self, epoch: int, max_epochs: int):
        self.epoch = epoch
        self.max_epochs = max_epochs

    def _loss_margin_scheduler(self):
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
        Z: Tuple[T, ...],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        loss_margin = self.config.loss_margin

        if self.config.loss_margin_scheduler:
            loss_margin = self._loss_margin_scheduler()
            self.loss_fn.loss_fn.margin = loss_margin

        loss = self.loss_fn(Z)

        self.log_step_metrics(
            step,
            {
                "train/loss": loss,
                "train/margin": loss_margin,
            },
        )

        return loss
