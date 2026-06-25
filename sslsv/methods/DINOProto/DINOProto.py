from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

import numpy as np

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMomentumMethod import (
    BaseMomentumMethod,
    BaseMomentumMethodConfig,
    initialize_momentum_params,
)

from sslsv.utils.distributed import get_world_size

from .DINOProtoLoss import DINOProtoLoss


class DINOProtoHead(nn.Module):
    """
    Head module for DINOProto.

    Attributes:
        mlp (nn.Sequential): MLP module.
        last_layer (nn.utils.weight_norm): Last layer module.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        """
        Initialize a DINOProto head module.

        Args:
            input_dim (int): Dimension of the input.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output.

        Returns:
            None
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize weights.

        Args:
            m (nn.Module): PyTorch module.

        Returns:
            None
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: T) -> T:
        """
        Forward pass.

        Args:
            x (T): Input tensor.

        Returns:
            T: Output tensor.
        """
        x = self.head(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


@dataclass
class DINOProtoConfig(BaseMomentumMethodConfig):
    """
    DINOProto method configuration.

    Attributes:
        global_count (int): Number of global (large) views.
        local_count (int): Number of local (small) views.
        start_tau (float): Initial value for tau (momentum parameters update).
        head_hidden_dim (int): Head hidden dimension.
        head_output_dim (int): Head output dimension.
        nb_prototypes (int): Number of prototypes.
        student_temperature (float): Temperature value for the student.
        teacher_temperature (float): Temperature value for the teacher.
        memax_weight (float): Weight for Me-Max regularization.
        koleo_weight (float): Weight for Koleo regularization.
    """

    global_count: int = 1
    local_count: int = 4

    start_tau: float = 0.996

    head_hidden_dim: int = 2048
    head_output_dim: int = 256

    nb_prototypes: int = 1024

    student_temperature: float = 0.1
    teacher_temperature: float = 0.025

    memax_weight: float = 1.0
    koleo_weight: float = 0.1


class DINOProto(BaseMomentumMethod):
    """
    DINOProto method.

    Attributes:
        head (DINOProtoHead): Head module.
        head_momentum (DINOProtoHead): Head momentum module.
        loss_fn (DINOProtoLoss): Loss function.
    """

    def __init__(
        self,
        config: DINOProtoConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a DINOProto method.

        Args:
            config (DINOProtoConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.head = DINOProtoHead(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            output_dim=config.head_output_dim,
        )

        self.head_momentum = DINOProtoHead(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            output_dim=config.head_output_dim,
        )
        initialize_momentum_params(self.head, self.head_momentum)

        # Prototypes
        _sqrt_k = (1. / config.head_output_dim) ** 0.5
        self.prototypes = nn.Parameter(torch.empty(config.nb_prototypes, config.head_output_dim))
        nn.init.uniform_(self.prototypes, -_sqrt_k, _sqrt_k)

        self.loss_fn = DINOProtoLoss(
            global_count=config.global_count,
            local_count=config.local_count,
            nb_prototypes=config.nb_prototypes,
            student_temp=config.student_temperature,
            teacher_temp=config.teacher_temperature,
            memax_weight=config.memax_weight,
            koleo_weight=config.koleo_weight,
        )

    def forward(self, X: T, training: bool = False) -> Union[T, Tuple[T, T, T]]:
        """
        Forward pass.

        Args:
            X (T): Input tensor.
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            Union[T, Tuple[T, T, T]]: Encoder output for inference or embeddings for training.
        """
        if not training:
            return self.encoder_momentum(X)

        N, V, L = X.size()

        X = X.transpose(0, 1)

        global_frames = X[:self.config.global_count, :, :].reshape(-1, L)
        local_frames = X[self.config.global_count:, :, : L // 2].reshape(-1, L // 2)

        T = self.head_momentum(self.encoder_momentum(global_frames))

        Y = self.encoder(local_frames)
        S = self.head(Y)

        return S, T, Y

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
        Update the learning rate for DINOProto method.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            init_lr (float): Initial learning rate from configuration.
            init_wd (float): Initial weight decay from configuration.
            step (int): Current training step.
            nb_steps (int): Total number of training steps.
            nb_steps_per_epoch (int): Number of training steps per epoch.

        Returns:
            Tuple[float, float]: Updated learning rate and initial weight decay.
        """
        lr, wd = super().update_optim(
            optimizer,
            init_lr,
            init_wd,
            step,
            nb_steps,
            nb_steps_per_epoch,
        )

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr
            param_group["weight_decay"] = init_wd if i == 0 else 0

        return lr, init_wd

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = [{"params": self.head.parameters()}]
        params = super().get_learnable_params() + extra_learnable_params

        # Do not apply weight decay on biases and norms parameters
        regularized = []
        not_regularized = []
        for module in params:
            for param in module["params"]:
                if not param.requires_grad:
                    continue

                if len(param.shape) == 1:
                    not_regularized.append(param)
                else:
                    regularized.append(param)

        return [
            {"params": regularized},
            {"params": not_regularized},
            {"params": [self.prototypes]}
        ]

    def get_momentum_pairs(self) -> List[Tuple[nn.Module, nn.Module]]:
        """
        Get a list of modules and their associated momentum module.

        Returns:
            List[Tuple[nn.Module, nn.Module]]: List of (module, module_momentum) pairs.
        """
        extra_momentum_pairs = [(self.head, self.head_momentum)]
        return super().get_momentum_pairs() + extra_momentum_pairs

    def train_step(
        self,
        Z: Tuple[T, T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (Tuple[T, T, T]): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        S, T, Y = Z
        
        loss, loss_metrics = self.loss_fn(S, T, self.prototypes, Y)

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/tau": self.momentum_updater.tau,
                **loss_metrics
            },
        )

        return loss