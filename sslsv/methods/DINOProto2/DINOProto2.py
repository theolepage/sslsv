from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

import numpy as np

from sslsv.utils.distributed import gather

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMomentumMethod import (
    BaseMomentumMethod,
    BaseMomentumMethodConfig,
    initialize_momentum_params,
)

from sslsv.methods._OSPS.OSPS import OSPS, OSPSConfig

from .DINOProto2Loss import DINOProto2Loss


class DINOProto2Head(nn.Module):
    """
    Head module for DINOProto2.

    Attributes:
        head (nn.Sequential): MLP module.
        prototypes (nn.utils.weight_norm): Last layer module.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nb_prototypes: int,
    ):
        """
        Initialize a DINOProto2 head module.

        Args:
            input_dim (int): Dimension of the input.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output.
            nb_prototypes (int): Number of prototypes.

        Returns:
            None
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.prototypes = nn.utils.weight_norm(
            nn.Linear(output_dim, nb_prototypes, bias=False)
        )
        self.prototypes.weight_g.data.fill_(1)
        self.prototypes.weight_g.requires_grad = False

    def forward(self, x: T) -> Tuple[T, T]:
        """
        Forward pass.

        Args:
            x (T): Input tensor.

        Returns:
            Tuple[T, T]: Output tensor and embedding it was determined from, taken before
                it is normalized so that the dimensions can be regularized on their own
                scale.
        """
        E = self.head(x)
        x = F.normalize(E, p=2, dim=-1)
        return self.prototypes(x), E


@dataclass
class DINOProto2Config(BaseMomentumMethodConfig):
    """
    DINOProto2 method configuration.

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
        fronorm_weight (float): Weight of the Frobenius dimension regularization, which
            decorrelates the dimensions of the embeddings of the head by minimizing the
            log of the Frobenius norm of their correlation matrix. None disables it.
            See "Pushing the Frontiers of Self-Distillation Prototypes Network with
            Dimension Regularization and Score Normalization" (Chen et al., 2025).
        osps (OSPSConfig): Online Self-Supervised Positive Sampling (OSPS) configuration.
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

    fronorm_weight: Optional[float] = None

    osps: OSPSConfig = None


class DINOProto2(BaseMomentumMethod):
    """
    DINOProto2 method: DINOProto with an optional OSPS module.

    OSPS samples pseudo-positives as SSPS does, but its memory queues and its
    epoch-boundary K-Means are replaced by a codebook maintained by online K-Means and by
    a small memory of embeddings per cluster (see OSPS).

    Attributes:
        head (DINOProto2Head): Head module.
        head_momentum (DINOProto2Head): Head momentum module.
        loss_fn (DINOProto2Loss): Loss function.
        osps (Optional[OSPS]): Online Self-Supervised Positive Sampling module.
    """

    def __init__(
        self,
        config: DINOProto2Config,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a DINOProto2 method.

        Args:
            config (DINOProto2Config): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.head = DINOProto2Head(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            output_dim=config.head_output_dim,
            nb_prototypes=config.nb_prototypes,
        )

        self.head_momentum = DINOProto2Head(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            output_dim=config.head_output_dim,
            nb_prototypes=config.nb_prototypes,
        )
        initialize_momentum_params(self.head, self.head_momentum)

        self.loss_fn = DINOProto2Loss(
            global_count=config.global_count,
            local_count=config.local_count,
            student_temp=config.student_temperature,
            teacher_temp=config.teacher_temperature,
            memax_weight=config.memax_weight,
            koleo_weight=config.koleo_weight,
        )

        self.osps = (
            OSPS(config.osps, self.encoder.encoder_dim, config.nb_prototypes)
            if config.osps
            else None
        )

    def forward(
        self, X: T, training: bool = False
    ) -> Union[T, Tuple[T, T, T, T, T, T]]:
        """
        Forward pass.

        Args:
            X (T): Input tensor.
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            Union[T, Tuple[T, T, T, T, T, T]]: Encoder output for inference or embeddings
                for training.
        """
        if not training:
            return self.encoder_momentum(X)

        N, V, L = X.size()

        X = X.transpose(0, 1)

        global_frames = X[:self.config.global_count, :, :].reshape(-1, L)
        local_frames = X[self.config.global_count:, :, : L // 2].reshape(-1, L // 2)

        Y_global = self.encoder_momentum(global_frames)
        T, E_teacher = self.head_momentum(Y_global)

        Y = self.encoder(local_frames)
        S, E_student = self.head(Y)

        # Global frames are neither augmented nor cropped and are thus directly used as
        # reference representations for OSPS (no extra frame required).
        Y_ref = None
        if self.osps:
            Y_ref = F.normalize(Y_global[:N].detach(), p=2, dim=-1)

        return S, T, Y, Y_ref, E_student, E_teacher

    @staticmethod
    def fronorm(E: T) -> T:
        """
        Determine the Frobenius dimension regularization of a batch of embeddings.

        The correlation between every pair of dimensions is determined over the batch and
        the log of the Frobenius norm of the resulting matrix is minimized, which drives
        the dimensions of the embeddings to carry information independently. The diagonal
        of the matrix is made of ones and thus only offsets the term by a constant.

        Args:
            E (T): Embeddings of the head. Shape: (N, D).

        Returns:
            T: Regularization term.
        """
        # The correlation is estimated over the whole batch: a single process holds
        # fewer embeddings than they have dimensions, which would inflate every
        # correlation it measures
        E = gather(E).float()

        norms = E.pow(2).sum(dim=0).clamp(min=1e-8).sqrt()
        C = (E.T @ E) / torch.outer(norms, norms)

        # The correlation is not centered, so the diagonal is exactly one and the term
        # cannot be lowered by driving a dimension constant. Holding it at D makes that
        # explicit and cancels its gradient, as the paper does.
        D = C.size(0)
        off_diagonal = C.pow(2).sum() - C.diagonal().pow(2).sum()

        return 0.5 * torch.log(D + off_diagonal)

    @staticmethod
    @torch.no_grad()
    def alive(E: T) -> float:
        """
        Determine the fraction of dimensions that are not constant over the batch.

        Args:
            E (T): Embeddings of the head. Shape: (N, D).

        Returns:
            float: Fraction of the dimensions that still vary.
        """
        E = gather(E).float()
        return (E.std(dim=0) > 1e-4).float().mean().item()

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
        Update the learning rate for DINOProto2 method.

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
        Z: Tuple[T, T, T, T, T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (Tuple[T, T, T, T, T, T]): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        S, T, Y, Y_ref, E_student, E_teacher = Z

        osps_metrics = {}
        if self.osps:
            T = self.osps.substitute(indices, T, Y_ref)

        loss, loss_metrics = self.loss_fn(S, T, Y)

        fronorm_metrics = {}
        if self.config.fronorm_weight is not None:
            fronorm = self.fronorm(E_student) + self.fronorm(E_teacher)
            loss = loss + self.config.fronorm_weight * fronorm
            fronorm_metrics = {
                "train/fronorm": fronorm.item(),
                # Fraction of the dimensions that still vary over the batch: the term
                # reaches its floor both when they are decorrelated and when they are dead
                "train/fronorm_alive": self.alive(E_student),
            }

        if self.osps:
            loss = loss + self.osps.repel(Y)
            osps_metrics = self.osps.step_metrics

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/tau": self.momentum_updater.tau,
                **loss_metrics,
                **fronorm_metrics,
                **osps_metrics,
            },
        )

        return loss

    def on_train_start(self):
        """
        Initialize OSPS and the AAM head.

        Returns:
            None
        """
        super().on_train_start()

        dataset_size = len(self.trainer.train_dataloader.dataset)
        train_csv = (
            self.trainer.config.dataset.base_path / self.trainer.config.dataset.train
        )

        if self.osps:
            self.osps.initialize(
                dataset_size=dataset_size,
                train_csv=train_csv,
                device=self.trainer.device,
            )

    def on_train_epoch_start(self, epoch: int, max_epochs: int):
        """
        Enable OSPS and determine its clustering metrics.

        Args:
            epoch (int): Current epoch.
            max_epochs (int): Total number of epochs.

        Returns:
            None
        """
        super().on_train_epoch_start(epoch, max_epochs)

        if self.osps:
            self.osps.set_epoch(epoch)
            self.osps.compute_metrics()
