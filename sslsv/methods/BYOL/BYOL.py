from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from torch import nn
from torch import Tensor as T

from dataclasses import dataclass

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMomentumMethod import (
    BaseMomentumMethod,
    BaseMomentumMethodConfig,
    initialize_momentum_params,
)

from .BYOLLoss import BYOLLoss


@dataclass
class BYOLConfig(BaseMomentumMethodConfig):
    """
    BYOL (Bootstrap Your Own Latent) method configuration.

    Attributes:
        projector_hidden_dim (int): Hidden dimension of the projector.
        projector_output_dim (int): Output dimension of the projector.
        predictor_hidden_dim (int): Hidden dimension of the predictor.
    """

    start_tau: float = 0.996

    projector_hidden_dim: int = 4096
    projector_output_dim: int = 256

    predictor_hidden_dim: int = 4096


class BYOL(BaseMomentumMethod):
    """
    BYOL (Bootstrap Your Own Latent) method.

    Paper:
        Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
        *Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond,
        Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar,
        Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko*
        NeurIPS 2020
        https://arxiv.org/abs/2006.07733

    Attributes:
        projector (nn.Module): Projector module.
        projector_momentum (nn.Module): Projector momentum module.
        predictor (nn.Module): Predictor module.
        loss_fn (BYOLLoss): Loss function.
    """

    def __init__(
        self,
        config: BYOLConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a BYOL method.

        Args:
            config (BYOLConfig): Method configuration.
            create_encoder_fn (Callable): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
        )

        self.projector_momentum = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
        )
        initialize_momentum_params(self.projector, self.projector_momentum)

        self.predictor = nn.Sequential(
            nn.Linear(config.projector_output_dim, config.predictor_hidden_dim),
            nn.BatchNorm1d(config.predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.predictor_hidden_dim, config.projector_output_dim),
        )

        self.loss_fn = BYOLLoss()

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

        P_1 = self.predictor(self.projector(self.encoder(X_1)))
        Z_1 = self.projector_momentum(self.encoder_momentum(X_1))

        P_2 = self.predictor(self.projector(self.encoder(X_2)))
        Z_2 = self.projector_momentum(self.encoder_momentum(X_2))

        return Z_1, Z_2, P_1, P_2

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ]
        return super().get_learnable_params() + extra_learnable_params

    def get_momentum_pairs(self) -> List[Tuple[nn.Module, nn.Module]]:
        """
        Get a list of modules and their associated momentum module.

        Returns:
            List[Tuple[nn.Module, nn.Module]]: List of (module, module_momentum) pairs.
        """
        extra_momentum_pairs = [(self.projector, self.projector_momentum)]
        return super().get_momentum_pairs() + extra_momentum_pairs

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

        loss = self.loss_fn(P_1, Z_2) + self.loss_fn(P_2, Z_1)

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/tau": self.momentum_updater.tau,
            },
        )

        return loss
