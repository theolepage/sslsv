from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMomentumMethod import (
    BaseMomentumMethod,
    BaseMomentumMethodConfig,
    initialize_momentum_params,
)

from .MoCoLoss import MoCoLoss

import torch.distributed as dist
from sslsv.utils.distributed import (
    gather,
    get_rank,
    get_world_size,
    is_dist_initialized,
)


@dataclass
class MoCoConfig(BaseMomentumMethodConfig):
    """
    MoCo method configuration.

    Attributes:
        start_tau (float): Initial value for tau (momentum parameters update).
        tau_scheduler (bool): Whether to use a cosine scheduler for tau (momentum parameters update).
        temperature (float): Temperature value.
        queue_size (int): Size of the queue.
        enable_projector (bool): Whether to enable the projector.
        projector_hidden_dim (int): Hidden dimension of the projector.
        projector_output_dim (int): Output dimension of the projector.
        prevent_class_collisions (bool): Whether to prevent class collisions using labels.
    """

    start_tau: float = 0.999
    tau_scheduler: bool = False

    temperature: float = 0.2

    queue_size: int = 65536

    enable_projector: bool = True
    projector_hidden_dim: int = 2048
    projector_output_dim: int = 128

    prevent_class_collisions: bool = False


class MoCo(BaseMomentumMethod):
    """
    MoCo (MOmentum COntrastive Learning) v2+ method.

    Paper:
        Improved Baselines with Momentum Contrastive Learning
        *Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He*
        arXiv preprint 2020
        https://arxiv.org/abs/2003.04297

    Attributes:
        queue_size (int): Size of the queue.
        queue (T): Buffer to store embeddings to use as negatives.
        queue_ptr (T): Pointer for the queue buffer.
        queue_labels (T): Buffer to store labels for preventing class collisions.
        queue_labels_ptr (T): Pointer for the queue_labels buffer.
        projector (nn.Sequential): Projector module.
        projector_momentum (nn.Sequential): Projector momentum module.
        loss_fn (MoCoLoss): Loss function.
    """

    SSPS_NB_POS_EMBEDDINGS = 2

    def __init__(
        self,
        config: MoCoConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        """
        Initialize a MoCo method.

        Args:
            config (MoCoConfig): Method configuration.
            create_encoder_fn (Callable[[], BaseEncoder]): Function that creates an encoder object.

        Returns:
            None
        """
        super().__init__(config, create_encoder_fn)

        self.queue_size = config.queue_size

        self.embeddings_dim = self.encoder.encoder_dim

        if config.enable_projector:
            self.embeddings_dim = config.projector_output_dim

            # self.projector = nn.Sequential(
            #     nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            # )
            self.projector = nn.Sequential(
                nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            )

            # self.projector_momentum = nn.Sequential(
            #     nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            # )
            self.projector_momentum = nn.Sequential(
                nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_hidden_dim),
                nn.BatchNorm1d(config.projector_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
            )
            initialize_momentum_params(self.projector, self.projector_momentum)

        self.register_buffer(
            "queue", torch.randn(2, self.embeddings_dim, self.queue_size)
        )
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if config.prevent_class_collisions:
            self.register_buffer("queue_labels", torch.zeros(self.queue_size))
            self.register_buffer("queue_labels_ptr", torch.zeros(1, dtype=torch.long))

        self.loss_fn = MoCoLoss(config.temperature)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: T) -> Tuple[T, T]:
        """
        Batch shuffle to prevent leaking information through the BNs.

        Args:
            x (T): Input tensor.

        Returns:
            Tuple[T, T]: Shuffled input tensor and the corresponding unshuffle indices.
        """
        x_gather = gather(x)

        idx_shuffle = torch.randperm(
            x_gather.size(0),
            device=get_rank() if is_dist_initialized() else None
        )
        if is_dist_initialized():
            dist.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)

        idx_this = idx_shuffle.view(get_world_size(), -1)[get_rank()]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: T, idx_unshuffle: T) -> T:
        """
        Batch unshuffle.

        Args:
            x (T): Input tensor.
            idx_unshuffle (T): Indices used for unshuffling.

        Returns:
            T: Unshuffled input tensor.
        """
        x_gather = gather(x)

        idx_this = idx_unshuffle.view(get_world_size(), -1)[get_rank()]

        return x_gather[idx_this]

    def _compute_embeddings(self, X: T, momentum: bool = False) -> T:
        """
        Compute embeddings for training.

        Args:
            X (T): Input tensor.
            momentum (bool): Whether to use momentum modules. Defaults to False.

        Returns:
            T: Output tensor.
        """
        if not momentum:
            if self.config.enable_projector:
                return self.projector(self.encoder(X))
            return self.encoder(X)
        else:
            if self.config.enable_projector:
                return self.projector_momentum(self.encoder_momentum(X))
            return self.encoder_momentum(X)

    def forward(self, X: T, training: bool = False) -> Union[T, Tuple[T, T, T, T, T]]:
        """
        Forward pass.

        Args:
            X (T): Input tensor.
            training (bool): Whether the forward pass is for training. Defaults to False.

        Returns:
            Union[T, Tuple[T, T, T, T, T]]: Encoder output for inference or embeddings for training.
        """
        if not training:
            return self.encoder(X)

        frame_length = self.trainer.config.dataset.frame_length

        # Queries
        Q_1 = self._compute_embeddings(X[:, 0, :frame_length])
        Q_2 = self._compute_embeddings(X[:, 1, :frame_length])

        # Keys
        X_s, idx_unshuffle = self._batch_shuffle_ddp(X)
        K_1 = self._compute_embeddings(X_s[:, 0, :frame_length], momentum=True)
        K_2 = self._compute_embeddings(X_s[:, 1, :frame_length], momentum=True)
        K_1 = self._batch_unshuffle_ddp(K_1, idx_unshuffle)
        K_2 = self._batch_unshuffle_ddp(K_2, idx_unshuffle)

        Y_ref = None
        if self.ssps:
            encoder_training_mode = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                Y_ref = F.normalize(self.encoder(X[:, -1]).detach(), p=2, dim=-1)
            if encoder_training_mode:
                self.encoder.train()

        return Q_1, K_2, Q_2, K_1, Y_ref

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

    def get_momentum_pairs(self) -> List[Tuple[nn.Module, nn.Module]]:
        """
        Get a list of modules and their associated momentum module.

        Returns:
            List[Tuple[nn.Module, nn.Module]]: List of (module, module_momentum) pairs.
        """
        extra_momentum_pairs = []
        if self.config.enable_projector:
            extra_momentum_pairs = [(self.projector, self.projector_momentum)]
        return super().get_momentum_pairs() + extra_momentum_pairs

    @torch.no_grad()
    def _enqueue(self, keys: T):
        """
        Enqueue keys into queue.

        Args:
            keys (T): Tensor of embeddings (Keys).

        Returns:
            None

        Raises:
            AssertionError: If the queue_size is not divisible by the batch size.
        """
        batch_size = keys.size(1)

        assert self.queue_size % batch_size == 0

        ptr = int(self.queue_ptr)
        self.queue[:, :, ptr : ptr + batch_size] = keys.permute(0, 2, 1)

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    @torch.no_grad()
    def _enqueue_labels(self, labels: T):
        """
        Enqueue labels into queue_labels.

        Args:
            labels (T): Tensor of labels.

        Returns:
            None

        Raises:
            AssertionError: If the queue size is not divisible by the batch size.
        """
        batch_size = labels.size(0)

        assert self.queue_size % batch_size == 0

        ptr = int(self.queue_labels_ptr)
        self.queue_labels[ptr : ptr + batch_size] = labels

        self.queue_labels_ptr[0] = (ptr + batch_size) % self.queue_size

    def train_step(
        self,
        Z: Tuple[T, T, T, T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        """
        Perform a training step.

        Args:
            Z (Tuple[T, T, T, T, T]): Embedding tensors.
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

        if self.ssps:
            self.ssps.sample(indices, Y_ref)
            K_1_pp = self.ssps.apply(0, K_1)
            K_2_pp = self.ssps.apply(1, K_2)
            self.ssps.update_buffers(step_rel, indices, Y_ref, [K_1, K_2])
            loss_1, metrics_1 = self.loss_fn(Q_1, K_2_pp, queue[1], current_labels, queue_labels)
            loss_2, metrics_2 = self.loss_fn(Q_2, K_1_pp, queue[0], current_labels, queue_labels)
            loss = (loss_1 + loss_2) / 2

            self._enqueue(torch.stack((gather(K_1_pp), gather(K_2_pp))))
        else:
            loss_1, metrics_1 = self.loss_fn(Q_1, K_2, queue[1], current_labels, queue_labels)
            loss_2, metrics_2 = self.loss_fn(Q_2, K_1, queue[0], current_labels, queue_labels)
            loss = (loss_1 + loss_2) / 2

            self._enqueue(torch.stack((gather(K_1), gather(K_2))))

        if self.config.prevent_class_collisions:
            self._enqueue_labels(gather(labels))

        loss_metrics = {k:(metrics_1[k] + metrics_2[k]) / 2 for k in metrics_1} 

        self.log_step_metrics(
            {
                "train/loss": loss,
                "train/tau": self.momentum_updater.tau,
                **loss_metrics
            },
        )

        return loss
