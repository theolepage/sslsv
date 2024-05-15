from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .SimSiamLoss import SimSiamLoss


@dataclass
class SimSiamConfig(BaseMethodConfig):
    """
    SimSiam method configuration.

    Attributes:
        projector_hidden_dim (int): Hidden dimension of the projector.
        projector_output_dim (int): Output dimension of the projector.
        pred_hidden_dim (int): Hidden dimension of the predictor.
    """

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 2048

    pred_hidden_dim: int = 512


class SimSiam(BaseMethod):
    """
    SimSiam (Simple Siamese Representation Learning) method.

    Paper:
        Exploring Simple Siamese Representation Learning
        *Xinlei Chen, Kaiming He*
        CVPR 2021
        https://arxiv.org/abs/2011.10566

    Attributes:
        projector (nn.Sequential): Projector module.
        predictor (nn.Sequential): Predictor module.
        loss_fn (SimSiamLoss): Loss function.
    """

    def __init__(
        self,
        config: SimSiamConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a SimSiam method.

        Args:
            config (SimSiamConfig): Method configuration.
            create_encoder_fn (Callable): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(
                self.encoder.encoder_dim,
                config.projector_hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                config.projector_hidden_dim,
                config.projector_hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            nn.BatchNorm1d(config.projector_output_dim, affine=False),
        )
        # hack: not use bias as it is followed by BN
        self.projector[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(config.projector_output_dim, config.pred_hidden_dim, bias=False),
            nn.BatchNorm1d(config.pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.pred_hidden_dim, config.projector_output_dim),
        )

        self.loss_fn = SimSiamLoss()

    def forward(self, X: T, training: bool = False) -> Union[T, Tuple[T, T, T, T]]:
        """
        Forward pass.

        Args:
            X (T): Input tensor.
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            Union[T, Tuple[T, T, T, T]]: Encoder output for inference or embeddings for training.
        """
        if not training:
            return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Z_1 = self.projector(self.encoder(X_1))
        P_1 = self.predictor(Z_1)

        Z_2 = self.projector(self.encoder(X_2))
        P_2 = self.predictor(Z_2)

        return Z_1, Z_2, P_1, P_2

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters(), "fix_lr": True},
        ]
        return super().get_learnable_params() + extra_learnable_params

    def update_optim(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        init_wd: float,
        step: int,
        nb_steps: int,
        nb_steps_per_epoch: int,
    ) -> Tuple[float, float]:
        """
        Do not update the learning rate for the predictor.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            init_lr (float): Initial learning rate from configuration.
            init_wd (float): Initial weight decay from configuration.
            step (int): Current training step.
            nb_steps (int): Total number of training steps.
            nb_steps_per_epoch (int): Number of training steps per epoch.

        Returns:
            Tuple[float, float]: Learning rate and weight decay.
        """
        lr, wd = super().update_optim(
            optimizer,
            init_lr,
            init_wd,
            step,
            nb_steps,
            nb_steps_per_epoch,
        )

        for param_group in optimizer.param_groups:
            if "fix_lr" in param_group and param_group["fix_lr"]:
                param_group["lr"] = init_lr

        return lr, wd

    def train_step(
        self,
        Z: Tuple[T, T, T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (Tuple[T, T, T, T]): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        Z_1, Z_2, P_1, P_2 = Z

        loss = (self.loss_fn(P_1, Z_2) + self.loss_fn(P_2, Z_1)) / 2

        z1_std = F.normalize(Z_1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(Z_2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/z_std": z_std.detach(),
            },
        )

        return loss
