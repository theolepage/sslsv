from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods.MoCo.MoCo import MoCo, MoCoConfig

from .MoCoMarginsLoss import MoCoMarginsLoss, MoCoMarginsLossConfig

from sslsv.utils.distributed import gather


@dataclass
class MoCoMarginsConfig(MoCoConfig):
    """
    MoCo Margins method configuration.

    Attributes:
        loss (MoCoMarginsLossConfig): Loss configuration.
    """

    loss: MoCoMarginsLossConfig = None


class MoCoMargins(MoCo):
    """
    MoCo Margins method.

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
        loss_fn (MoCoMarginsLoss): Loss function.
    """

    def __init__(
        self,
        config: MoCoMarginsConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a MoCo Margins method.

        Args:
            config (MoCoMarginsConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.epoch = 0
        self.max_epochs = 0

        self.loss_fn = MoCoMarginsLoss(config.loss)

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
        Q_1, K_2, Q_2, K_1, Y_ref = Z

        Q_1 = F.normalize(Q_1, p=2, dim=1)
        K_2 = F.normalize(K_2, p=2, dim=1)
        Q_2 = F.normalize(Q_2, p=2, dim=1)
        K_1 = F.normalize(K_1, p=2, dim=1)

        queue = self.queue.clone().detach()

        current_labels = None
        queue_labels = None
        if self.config.prevent_class_collisions:
            current_labels = labels
            queue_labels = self.queue_labels.clone().detach()

        margin = self.loss_fn.update_margin(self.epoch, self.max_epochs)

        if self.ssps:
            self.ssps.sample(indices, Y_ref)
            K_1_pp = self.ssps.apply(0, K_1)
            K_2_pp = self.ssps.apply(1, K_2)
            self.ssps.update_buffers(step_rel, indices, Y_ref, [K_1, K_2])
            loss = (
                self.loss_fn(Q_1, K_2_pp, queue[1], current_labels, queue_labels)
                + self.loss_fn(Q_2, K_1_pp, queue[0], current_labels, queue_labels)
            ) / 2

            self._enqueue(torch.stack((gather(K_1_pp), gather(K_2_pp))))
        else:
            loss = (
                self.loss_fn(Q_1, K_2, queue[1], current_labels, queue_labels)
                + self.loss_fn(Q_2, K_1, queue[0], current_labels, queue_labels)
            ) / 2

            self._enqueue(torch.stack((gather(K_1), gather(K_2))))

        if self.config.prevent_class_collisions:
            self._enqueue_labels(gather(labels))

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/margin": margin,
                "train/tau": self.momentum_updater.tau,
            },
        )

        return loss
