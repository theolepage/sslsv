from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseSiameseMethod import BaseSiameseMethod, BaseSiameseMethodConfig

from sslsv.methods.CPC.InfoNCELoss import InfoNCELoss
from sslsv.methods.VICReg.VICRegLoss import VICRegLoss
from sslsv.methods.BarlowTwins.BarlowTwinsLoss import BarlowTwinsLoss


class LossTypeCombinerEnum(Enum):
    """
    Enumeration representing different loss types for Combiner method.

    Members:
        INFONCE (str): InfoNCE loss.
        VICREG (str): VICReg loss.
        BARLOWTWINS (str): Barlow Twins loss.
    """

    INFONCE = "infonce"
    VICREG = "vicreg"
    BARLOWTWINS = "barlowtwins"


@dataclass
class LossItemCombinerConfig:
    """
    Configuration for loss items in Combiner configuration.

    Attributes:
        type (LossTypeCombinerEnum): Loss type.
        weight (float): Loss weight. Defaults to 1.0.
    """

    type: LossTypeCombinerEnum
    weight: float = 1.0


@dataclass
class CombinerConfig(BaseSiameseMethodConfig):
    """
    Combiner method configuration.

    Attributes:
        Y_losses (List[LossItemCombinerConfig]): List of loss items applied on Y (representations).
        Z_losses (List[LossItemCombinerConfig]): List of loss items applied on Z (embeddings).
    """

    Y_losses: List[LossItemCombinerConfig] = None
    Z_losses: List[LossItemCombinerConfig] = None


class Combiner(BaseSiameseMethod):
    """
    Combiner method.

    Attributes:
        LOSS_FUNCTIONS (Dict): Dictionary mapping LossTypeCombinerEnum to corresponding loss functions.
    """

    LOSS_FUNCTIONS = {
        LossTypeCombinerEnum.INFONCE: InfoNCELoss(),
        LossTypeCombinerEnum.VICREG: VICRegLoss(),
        LossTypeCombinerEnum.BARLOWTWINS: BarlowTwinsLoss(),
    }

    def __init__(
        self,
        config: CombinerConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a Combiner method.

        Args:
            config (CombinerConfig): Method configuration.
            create_encoder_fn (callable): Function that creates an object object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

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

        Y_1 = self.encoder(X_1)
        Y_2 = self.encoder(X_2)

        Z_1 = self.projector(Y_1) if self.config.enable_projector else None
        Z_2 = self.projector(Y_2) if self.config.enable_projector else None

        return Y_1, Y_2, Z_1, Z_2

    def compute_loss(self, Z_1: T, Z_2: T, losses: List[LossItemCombinerConfig]) -> T:
        """
        Compute loss on embeddings.

        Args:
            Z_1 (T): Embedding tensor of first view.
            Z_2 (T): Embedding tensor of second view.
            losses (List[LossItemCombinerConfig]): List of loss items.

        Returns:
            T: Loss tensor.
        """
        loss = 0
        for l in losses:
            loss += l.weight * Combiner.LOSS_FUNCTIONS[l.type](Z_1, Z_2)
        return loss

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
        Y_1, Y_2, Z_1, Z_2 = Z

        loss = 0
        metrics = {}

        # Representations
        Y_loss = self.compute_loss(Y_1, Y_2, self.config.Y_losses)
        Y_accuracy = InfoNCELoss.determine_accuracy(Y_1, Y_2)
        metrics = {
            **metrics,
            "train/Y_loss": Y_loss,
            "train/Y_accuracy": Y_accuracy,
        }
        loss += Y_loss

        # Embeddings
        if self.config.enable_projector:
            Z_loss = self.compute_loss(Z_1, Z_2, self.config.Z_losses)
            Z_accuracy = InfoNCELoss.determine_accuracy(Z_1, Z_2)
            metrics = {
                **metrics,
                "train/Z_loss": Z_loss,
                "train/Z_accuracy": Z_accuracy,
            }
            loss += Z_loss

        self.log_step_metrics(
            {
                **metrics,
                "train/loss": loss,
            },
        )

        return loss
