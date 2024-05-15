from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .SinkhornKnopp import SinkhornKnopp
from .SwAVLoss import SwAVLoss

from sslsv.utils.distributed import get_world_size


@dataclass
class SwAVConfig(BaseMethodConfig):
    """
    SwAV method configuration.

    Attributes:
        temperature (float): Temperature value.
        nb_prototypes (int): Number of prototypes.
        sk_nb_iters (int): Number of Sinkhorn-Knopp iterations.
        sk_epsilon (float): Regularization hyper-parameter for Sinkhorn-Knopp algorithm.
        queue_size (int): Size of the queue for storing embeddings.
        queue_start_epoch (int): Epoch at which the queue is used.
        freeze_prototypes_epochs (int): Number of epochs to freeze prototypes.
        projector_hidden_dim (int): Hidden dimension of the projector network.
        projector_output_dim (int): Output dimension of the projector network.
    """

    temperature: float = 0.1

    nb_prototypes: int = 3000

    sk_nb_iters: int = 3
    sk_epsilon: float = 0.05

    queue_size: int = 3840
    queue_start_epoch: int = 15

    freeze_prototypes_epochs: int = 1

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 128


class SwAV(BaseMethod):
    """
    SwAV (SWapping Assignments between multiple Views) method.

    Paper:
        Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
        *Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin*
        NeurIPS 2020
        https://arxiv.org/abs/2006.09882

    Attributes:
        epoch (int): Current training epoch.
        projector (nn.Sequential): Projector module.
        prototypes (nn.utils.weight_norm): Weight normalized linear layer for prototypes.
        sk (SinkhornKnopp): Sinkhorn-Knopp algorithm object.
        loss_fn (SwAVLoss): Loss function.
    """

    def __init__(
        self,
        config: SwAVConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a SwAV method.

        Args:
            config (SwAVConfig): Method configuration.
            create_encoder_fn (Callable): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.epoch = 0

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
        )

        self.prototypes = nn.utils.weight_norm(
            nn.Linear(
                config.projector_output_dim,
                config.nb_prototypes,
                bias=False,
            )
        )
        self.prototypes.weight_g.data.fill_(1)
        self.prototypes.weight_g.requires_grad = False

        self.sk = SinkhornKnopp(
            nb_iters=config.sk_nb_iters,
            epsilon=config.sk_epsilon,
        )

        self.loss_fn = SwAVLoss(config.temperature)

    def on_train_start(self):
        """
        Create queue to store training embeddings.

        Returns:
            None
        """
        if self.config.queue_size > 0:
            self.register_buffer(
                "queue",
                torch.zeros(
                    2,
                    self.config.queue_size // get_world_size(),
                    self.config.projector_output_dim,
                    device=self.trainer.device,
                ),
            )

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

        Z_1 = F.normalize(self.projector(self.encoder(X_1)), dim=-1)
        Z_2 = F.normalize(self.projector(self.encoder(X_2)), dim=-1)

        P_1 = self.prototypes(Z_1)
        P_2 = self.prototypes(Z_2)

        return Z_1, Z_2, P_1, P_2

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        """
        Get the learnable parameters.

        Returns:
            Iterable[Dict[str, Any]]: Collection of parameters.
        """
        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.prototypes.parameters()},
        ]
        return super().get_learnable_params() + extra_learnable_params

    def on_train_epoch_start(self, epoch: int, max_epochs: int):
        """
        Update training epoch value.

        Args:
            epoch (int): Current epoch.
            max_epochs (int): Total number of epochs.

        Returns:
            None
        """
        self.epoch = epoch

    def _get_sk_assignments(self, preds: List[T]) -> List[T]:
        """
        Get the assigned labels for a list of predictions using Sinkhorn-Knopp.

        Args:
            preds (List[T]): List of embeddings tensors.

        Returns:
            List[T]: List of assignments tensors.
        """
        N = preds[0].size(0)

        assignments = []

        use_queue = (
            self.config.queue_size > 0 and self.epoch >= self.config.queue_start_epoch
        )

        for i, P in enumerate(preds):
            if use_queue:
                P_queue = self.prototypes(self.queue[i])
                P = torch.cat((P, P_queue))
            assignments.append(self.sk(P)[:N])

        return assignments

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

        N, _ = Z_1.size()

        preds = [P_1, P_2]
        assignments = self._get_sk_assignments(preds)

        loss = self.loss_fn(preds, assignments)

        # Update queue
        if self.config.queue_size > 0:
            self.queue[:, N:] = self.queue[:, :-N].clone()
            self.queue[0, :N] = Z_1.detach()
            self.queue[1, :N] = Z_2.detach()

        self.log_step_metrics(
            {
                "train/loss": loss,
            },
        )

        return loss

    def on_after_backward(self):
        """
        Freeze prototypes.

        Returns:
            None
        """
        if self.epoch < self.config.freeze_prototypes_epochs:
            for p in self.prototypes.parameters():
                p.grad = None
