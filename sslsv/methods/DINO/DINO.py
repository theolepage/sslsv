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

from .DINOLoss import DINOLoss


class DINOHead(nn.Module):
    """
    Head module for DINO.

    Attributes:
        mlp (nn.Sequential): MLP module.
        last_layer (nn.utils.weight_norm): Last layer module.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        output_dim: int,
    ):
        """
        Initialize a DINO head module.

        Args:
            input_dim (int): Dimension of the input.
            hidden_dim (int): Dimension of the hidden layers.
            bottleneck_dim (int): Dimension of the bottleneck layer.
            output_dim (int): Dimension of the output.

        Returns:
            None
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

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
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)
        return x


@dataclass
class DINOConfig(BaseMomentumMethodConfig):
    """
    DINO method configuration.

    Attributes:
        start_tau (float): Initial value for tau (momentum parameters update).
        head_hidden_dim (int): Head hidden dimension.
        head_bottleneck_dim (int): Head bottleneck dimension.
        head_output_dim (int): Head output dimension.
        freeze_last_layer (int): Whether to freeze the last layer of the head.
        clip_grad (int): Clip gradients of student model.
        student_temperature (float): Temperature value for the student.
        teacher_temperature (float): Temperature value for the teacher.
        teacher_temperature_warmup (float): Initial temperature value for the teacher.
        teacher_temperature_warmup_epochs (int): Number of epochs for the teacher temperature warmup.
    """

    start_tau: float = 0.996

    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_output_dim: int = 65536

    freeze_last_layer: int = 1

    clip_grad: float = 3.0

    student_temperature: float = 0.1
    teacher_temperature: float = 0.04
    teacher_temperature_warmup: float = 0.04
    teacher_temperature_warmup_epochs: int = 10


class DINO(BaseMomentumMethod):
    """
    DINO (self-DIstillation with NO labels) method.

    Paper:
        Emerging Properties in Self-Supervised Vision Transformers
        *Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin*
        ICCV 2021
        https://arxiv.org/abs/2104.14294

    Attributes:
        current_epoch (int): Current training epoch.
        freeze_last_layer (bool): Whether to freeze the last layer of the head.
        head (DINOHead): Head module.
        head_momentum (DINOHead): Head momentum module.
        loss_fn (DINOLoss): Loss function.
    """

    SSPS_NB_POS_EMBEDDINGS = 2

    def __init__(
        self,
        config: DINOConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a DINO method.

        Args:
            config (DINOConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.current_epoch = 0

        self.freeze_last_layer = config.freeze_last_layer
        self.clip_grad = config.clip_grad

        self.embeddings_dim = config.head_output_dim

        self.head = DINOHead(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            output_dim=config.head_output_dim,
        )

        self.head_momentum = DINOHead(
            input_dim=self.encoder.encoder_dim,
            hidden_dim=config.head_hidden_dim,
            bottleneck_dim=config.head_bottleneck_dim,
            output_dim=config.head_output_dim,
        )
        initialize_momentum_params(self.head, self.head_momentum)

        self.loss_fn = DINOLoss(
            nb_prototypes=config.head_output_dim,
            student_temp=config.student_temperature,
            teacher_temp=config.teacher_temperature,
            teacher_temp_warmup=config.teacher_temperature_warmup,
            teacher_temp_warmup_epochs=config.teacher_temperature_warmup_epochs,
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

        global_frames = X[0:2, :, :].reshape(-1, L)
        local_frames = X[2:6, :, : L // 2].reshape(-1, L // 2)

        T = self.head_momentum(self.encoder_momentum(global_frames))

        S = torch.cat(
            (
                self.head(self.encoder(global_frames)),
                self.head(self.encoder(local_frames)),
            ),
            axis=0,
        )

        Y_ref = None
        if self.ssps:
            encoder_training_mode = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                Y_ref = F.normalize(self.encoder_momentum(X[-1]).detach(), p=2, dim=-1)
            if encoder_training_mode:
                self.encoder.train()

        return S, T, Y_ref

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
        Update the learning rate for DINO method.

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
        min_lr = 1e-5
        warmup_lr_schedule = np.linspace(0, init_lr, 10 * nb_steps_per_epoch)
        lr_schedule = min_lr + 0.5 * (init_lr - min_lr) * (
            1 + np.cos(np.pi * np.arange(nb_steps) / nb_steps)
        )
        lr_schedule = np.concatenate((warmup_lr_schedule, lr_schedule))
        lr = lr_schedule[step]

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

        return [{"params": regularized}, {"params": not_regularized}]

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
        S, T, Y_ref = Z

        if self.ssps:
            self.ssps.sample(indices, Y_ref)
            T_1, T_2 = T.chunk(2)
            T_1_pp = self.ssps.apply(0, T_1)
            T_2_pp = self.ssps.apply(1, T_2)
            T = torch.cat((T_1_pp, T_2_pp))
            self.ssps.update_buffers(step_rel, indices, Y_ref, [T_1, T_2])
            loss = self.loss_fn(S, T)
        else:
            loss = self.loss_fn(S, T)

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/tau": self.momentum_updater.tau,
            },
        )

        return loss

    def on_train_epoch_start(self, epoch: int, max_epochs: int):
        """
        Update training epoch value.

        Args:
            epoch (int): Current epoch.
            max_epochs (int): Total number of epochs.

        Returns:
            None
        """
        super().on_train_epoch_start(epoch, max_epochs)

        self.current_epoch = epoch
        self.loss_fn.epoch = epoch

    def on_after_backward(self):
        """
        Freeze last layer of head.

        Returns:
            None
        """
        for model in (self.encoder, self.head):
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    clip_coef = self.clip_grad / (param_norm + 1e-6)
                    if clip_coef < 1:
                        p.grad.data.mul_(clip_coef)

        if self.current_epoch < self.freeze_last_layer:
            for p in self.head.last_layer.parameters():
                p.grad = None
