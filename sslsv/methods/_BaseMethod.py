from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor as T

from dataclasses import dataclass

from sslsv.encoders._BaseEncoder import BaseEncoder


@dataclass
class BaseMethodConfig:

    pass


class BaseMethod(nn.Module):

    def __init__(
        self,
        config: BaseMethodConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        super().__init__()

        self.config = config

        self.trainer = None

        self.step_metrics = {}

        self.encoder = create_encoder_fn()

    def log_step_metrics(
        self,
        step: int,
        metrics: Dict[str, Union[T, int, float]],
    ):
        self.step_metrics[step] = metrics

    def forward(self, X: T, training: bool = False) -> T:
        return self.encoder(X)

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
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
        raise NotImplementedError

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_train_epoch_start(self, epoch: int, max_epochs: int):
        pass

    def on_train_epoch_end(self, epoch: int, max_epochs: int):
        pass

    def on_train_step_start(self, step: int, max_steps: int):
        pass

    def on_train_step_end(self, step: int, max_steps: int):
        pass

    def on_before_backward(self):
        pass

    def on_after_backward(self):
        pass
