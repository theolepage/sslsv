from dataclasses import dataclass
from typing import Callable, List, Tuple

import math

import torch
from torch import nn

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig


@torch.no_grad()
def initialize_momentum_params(module: nn.Module, module_momentum: nn.Module):
    """
    Initialize the parameters of a momentum module.

    * Copy the parameters from the module to the momentum module
    * Freeze the parameters of the momentum module

    Args:
        module (nn.Module): Main module.
        module_momentum (nn.Module): Momentum module.

    Returns:
        None
    """
    for p, p_m in zip(module.parameters(), module_momentum.parameters()):
        p_m.data.copy_(p.data)
        p_m.requires_grad = False


class MomentumUpdater:
    """
    Update the parameters of a momentum module with an EMA.

    Attributes:
        start_tau (float): Starting momentum coefficient.
        end_tau (float): Starting tau value.
        tau (float): Ending tau value.
    """

    def __init__(self, start_tau: float, end_tau: float):
        """
        Initiaize a momentum updater object.

        Args:
            start_tau (float): Starting tau value between 0 and 1.
            end_tau (float): Ending tau value between 0 and 1.

        Returns:
            None

        Raises:
            AssertionError: If the initial tau value is not in [0, 1].
            AssertionError: If the final tau value is not in [0, 1].
            AssertionError: If the final tau value is greater than the initial tau value.
        """
        super().__init__()

        assert 0 <= start_tau <= 1
        assert 0 <= end_tau <= 1 and start_tau <= end_tau

        self.start_tau = start_tau
        self.end_tau = end_tau

        self.tau = start_tau

    @torch.no_grad()
    def update(self, module: nn.Module, module_momentum: nn.Module):
        """
        Update the parameters of the momentum module.

        Args:
            module (nn.Module): Main module.
            module_momentum (nn.Module): Momentum module.

        Returns:
            None
        """
        for p, p_m in zip(module.parameters(), module_momentum.parameters()):
            p_m.data = self.tau * p_m.data + (1 - self.tau) * p.data

    def update_tau(self, step: int, max_steps: int):
        """
        Update the current momentum coefficient (tau).

        Args:
            step (int): Current training step.
            max_steps (int): Total number of training steps.

        Returns:
            None
        """
        self.tau = (
            self.end_tau
            - (self.end_tau - self.start_tau)
            * (math.cos(math.pi * step / max_steps) + 1)
            / 2
        )


@dataclass
class BaseMomentumMethodConfig(BaseMethodConfig):
    """
    Base configuration for momentum-based methods.

    Attributes:
        tau_scheduler (bool): Whether to use a cosine scheduler for tau (momentum parameters update).
        start_tau (float): Initial value for tau.
        end_tau (float): Final value for tau.
    """

    tau_scheduler: bool = True
    start_tau: float = 0.99
    end_tau: float = 1.0


class BaseMomentumMethod(BaseMethod):
    """
    Base class for momentum-based methods.

    Attributes:
        encoder_momentum (nn.Module): Momentum encoder.
        tau_scheduler (bool): Whether to use a tau scheduler for momentum parameters update.
        momentum_updater (MomentumUpdater): Momentum updater object.
    """

    def __init__(
        self,
        config: BaseMomentumMethodConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a momentum-based method.

        Args:
            config (BaseMomentumMethodConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.encoder_momentum = create_encoder_fn()
        initialize_momentum_params(self.encoder, self.encoder_momentum)

        self.tau_scheduler = config.tau_scheduler
        self.momentum_updater = MomentumUpdater(config.start_tau, config.end_tau)

    def get_momentum_pairs(self) -> List[Tuple[nn.Module, nn.Module]]:
        """
        Get a list of modules and their associated momentum module.

        Returns:
            List[Tuple[nn.Module, nn.Module]]: List of (module, module_momentum) pairs.
        """
        return [(self.encoder, self.encoder_momentum)]

    def on_train_step_start(self, step: int, max_steps: int):
        """
        Update tau.

        Args:
            step (int): Current training step.
            max_steps (int): Total number of training steps.

        Returns:
            None
        """
        if self.tau_scheduler:
            self.momentum_updater.update_tau(step, max_steps)

    def on_train_step_end(self, step: int, max_steps: int):
        """
        Update the momentum parameters.

        Args:
            step (int): Current training step.
            max_steps (int): Total number of training steps.

        Returns:
            None
        """
        for m, mm in self.get_momentum_pairs():
            self.momentum_updater.update(m, mm)
