from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor as T

from dataclasses import dataclass

from sslsv.encoders._BaseEncoder import BaseEncoder

from sslsv.methods._SSPS.SSPS import SSPS, SSPSConfig


@dataclass
class BaseMethodConfig:
    """
    Base configuration for methods.

    Attributes:
        ssps (SSPSConfig): Self-Supervised Positive Sampling (SSPS) configuration.
    """

    ssps: SSPSConfig = None


class BaseMethod(nn.Module):
    """
    Base class for methods.

    Attributes:
        config (BaseMethodConfig): Method configuration.
        trainer (Trainer): Trainer object.
        step_metrics (Dict[str, Union[T, int, float]]): Metrics for the current training step.
        encoder (BaseEncoder): Encoder object.
    """

    def __init__(
        self,
        config: BaseMethodConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a base method.

        Args:
            config (BaseMethodConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__()

        self.config = config

        self.trainer = None

        self.step_metrics = None

        self.encoder = create_encoder_fn()

        self.ssps = SSPS(config.ssps) if config.ssps else None
        self._ddp_params_and_buffers_to_ignore = [
            "ssps.queue_embeddings",
            "ssps.queue_indices",
        ]

    def log_step_metrics(
        self,
        metrics: Dict[str, Union[T, int, float]],
    ):
        """
        Store metrics for the current training step.

        Args:
            metrics (Dict[str, Union[T, int, float]]): Dictionary containing metrics.

        Returns:
            None
        """
        self.step_metrics = metrics

        if self.ssps:
            self.step_metrics.update(self.ssps.step_metrics)

    def forward(self, X: T, training: bool = False) -> T:
        """
        Forward pass.

        Args:
            X (T): Input tensor.
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            T: Output tensor.
        """
        return self.encoder(X)

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        return [{"params": self.encoder.parameters()}]

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
        Update the learning rate and weight decay of the optimizer.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            init_lr (float): Initial learning rate from configuration.
            init_wd (float): Initial weight decay from configuration.
            step (int): Current training step.
            nb_steps (int): Total number of training steps.
            nb_steps_per_epoch (int): Number of training steps per epoch.

        Returns:
            Tuple[float, float]: Updated learning rate and weight decay.
        """
        # Equivalent to StepLR(..., step_size=5, gamma=0.95)
        lr = init_lr * (0.95 ** ((step // nb_steps_per_epoch) // 5))

        # lr_schedule = (
        #    1e-4 + 0.5 * (init_lr - 1e-4) *
        #    (1 + np.cos(np.pi * np.arange(nb_steps) / nb_steps))
        # )
        # lr = lr_schedule[step]

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr, init_wd

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

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def on_train_start(self):
        """
        Perform actions at the start of the training.
        Initialize SSPS if enabled.

        Returns:
            None
        """
        if self.ssps:
            self.ssps.initialize(
                dataset_size=len(self.trainer.train_dataloader.dataset),
                batch_size=self.trainer.config.trainer.batch_size,
                embeddings_dim=self.encoder.encoder_dim,
                device=self.trainer.device,
            )

    def on_train_end(self):
        """
        Perform actions at the end of the training.

        Returns:
            None
        """
        pass

    def on_train_epoch_start(self, epoch: int, max_epochs: int):
        """
        Perform actions at the start of a training epoch.
        Prepare sampling for SSPS if enabled.

        Args:
            epoch (int): Current epoch.
            max_epochs (int): Total number of epochs.

        Returns:
            None
        """
        if self.ssps:
            self.ssps.set_epoch(epoch)
            self.ssps.prepare_sampling()

    def on_train_epoch_end(self, epoch: int, max_epochs: int):
        """
        Perform actions at the end of a training epoch.

        Args:
            epoch (int): Current epoch.
            max_epochs (int): Total number of epochs.

        Returns:
            None
        """
        pass

    def on_train_step_start(self, step: int, max_steps: int):
        """
        Perform actions at the start of a training step.

        Args:
            step (int): Current step.
            max_steps (int): Total number of steps.

        Returns:
            None
        """
        pass

    def on_train_step_end(self, step: int, max_steps: int):
        """
        Perform actions at the end of a training step.

        Args:
            step (int): Current step.
            max_steps (int): Total number of steps.

        Returns:
            None
        """
        pass

    def on_before_backward(self):
        """
        Perform actions before the backward pass (gradient computation).

        Returns:
            None
        """
        pass

    def on_after_backward(self):
        """
        Perform actions after the backward pass (gradient computation).

        Returns:
            None
        """
        pass
