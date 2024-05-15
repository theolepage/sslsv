from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from torch import nn
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig


@dataclass
class BaseSiameseMethodConfig(BaseMethodConfig):
    """
    Base configuration for siamese-based methods.

    Attributes:
        enable_projector (bool): Whether to enable the projector.
        projector_hidden_dim (int): Hidden dimension for the projector.
        projector_output_dim (int): Output dimension for the projector.
    """

    enable_projector: bool = True

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 2048


class BaseSiameseMethod(BaseMethod):
    """
    Base class for siamese-based methods.

    Attributes:
        projector (nn.Sequential): Projector module.
    """

    def __init__(
        self,
        config: BaseSiameseMethodConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a siamese-based method.

        Args:
            config (BaseSiameseMethodConfig): Method configuration.
            create_encoder_fn (Callable): Function that creates an encoder object.

        Returns:
            None.
        """
        super().__init__(config, create_encoder_fn)

        if config.enable_projector:
            self.projector = nn.Sequential(
                nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            )

    def forward(self, X: T, training: bool = False) -> Union[T, Tuple[T, T]]:
        """
        Forward pass.

        Args:
            X (T): Input tensor.
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            Union[T, Tuple[T, T]]: Encoder output for inference or embeddings for training.
        """
        if not training:
            return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Y_1 = self.encoder(X_1)
        Y_2 = self.encoder(X_2)

        Z_1 = self.projector(Y_1) if self.config.enable_projector else Y_1
        Z_2 = self.projector(Y_2) if self.config.enable_projector else Y_2

        return Z_1, Z_2

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = []
        if self.config.enable_projector:
            extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().get_learnable_params() + extra_learnable_params

    def train_step(
        self,
        Z: Tuple[T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (Tuple[T, T]): Embedding tensors.
            step (int): Current training step.
            step_rel (Optional[int]): Current training step (relative to the epoch).
            indices (Optional[T]): Training sample indices.
            labels (Optional[T]): Training sample labels.

        Returns:
            T: Loss tensor.
        """
        Z_1, Z_2 = Z

        loss = self.loss_fn(Z_1, Z_2)

        self.log_step_metrics(
            {
                "train/loss": loss,
            },
        )

        return loss
