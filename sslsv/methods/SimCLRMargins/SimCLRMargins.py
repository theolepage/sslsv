from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods.SimCLR.SimCLR import SimCLR, SimCLRConfig

from .SimCLRMarginsLoss import SimCLRMarginsLoss, SimCLRMarginsLossConfig


@dataclass
class SimCLRMarginsConfig(SimCLRConfig):
    """
    SimCLR Margins method configuration.

    Attributes:
        loss (SimCLRMarginsLossConfig): Loss configuration.
    """

    loss: SimCLRMarginsLossConfig = SimCLRMarginsLossConfig()


class SimCLRMargins(SimCLR):
    """
    SimCLR Margins method.

    Papers:
        - Experimenting with Additive Margins for Contrastive Self-Supervised Speaker Verification
          *Théo Lepage, Réda Dehak*
          https://arxiv.org/abs/2306.03664

        - Additive Margin in Contrastive Self-Supervised Frameworks to Learn Discriminative Speaker Representations
          *Théo Lepage, Réda Dehak*
          https://arxiv.org/abs/2404.14913

    Attributes:
        epoch (int): Current epoch.
        max_epochs (int): Maximum number of epochs.
        loss_fn (SimCLRMarginsLoss): Loss function.
    """

    def __init__(
        self,
        config: SimCLRMarginsConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a SimCLR Margins method.

        Args:
            config (SimCLRMarginsConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.epoch = 0
        self.max_epochs = 0

        self.loss_fn = SimCLRMarginsLoss(config.loss)

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = []
        if self.config.loss.margin_learnable:
            extra_learnable_params += [{"params": self.loss_fn.parameters()}]
        return super().get_learnable_params() + extra_learnable_params

    def on_train_epoch_start(self, epoch: int, max_epochs: int):
        """
        Update epoch and max_epochs values for margin scheduler.

        Args:
            epoch (int): Current epoch.
            max_epochs (int): Total number of epochs.

        Returns:
            None
        """
        super().on_train_epoch_start(epoch, max_epochs)

        self.epoch = epoch
        self.max_epochs = max_epochs

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
        Z_1, Z_2, Y_ref = Z

        margin = self.loss_fn.update_margin(self.epoch, self.max_epochs)

        if self.ssps:
            self.ssps.sample(indices, Y_ref)
            Z_2_pp = self.ssps.apply(0, Z_2)
            self.ssps.update_buffers(step_rel, indices, Y_ref, [Z_2])
            loss = self.loss_fn(
                Z_1,
                Z_2_pp,
                # ssps_assignments=self.ssps.sampling.assignments[indices],
            )
        else:
            loss = self.loss_fn(Z_1, Z_2)

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/margin": margin,
            },
        )

        return loss
