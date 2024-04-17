from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T

from sslsv.encoders._BaseEncoder import BaseEncoder
from sslsv.methods._BaseMethod import BaseMethod, BaseMethodConfig

from .KMeans import KMeans
from .DeepClusterLoss import DeepClusterLoss

from sslsv.utils.distributed import get_world_size


@dataclass
class DeepClusterConfig(BaseMethodConfig):

    temperature: float = 0.1

    nb_prototypes: Sequence[int] = field(default_factory=lambda: [3000, 3000, 3000])

    kmeans_nb_iters: int = 10

    projector_hidden_dim: int = 2048
    projector_output_dim: int = 128


class DeepCluster(BaseMethod):

    def __init__(
        self,
        config: DeepClusterConfig,
        create_encoder_fn: Callable[[], BaseEncoder],
    ):
        super().__init__(config, create_encoder_fn)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.encoder_dim, config.projector_hidden_dim),
            nn.BatchNorm1d(config.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projector_hidden_dim, config.projector_output_dim),
        )

        self.prototypes = nn.ModuleList(
            [
                nn.Linear(config.projector_output_dim, K, bias=False)
                for K in config.nb_prototypes
            ]
        )
        for layer in self.prototypes:
            for p in layer.parameters():
                p.requires_grad = False
            layer.weight.copy_(F.normalize(layer.weight.data.clone(), dim=-1))

        self.loss_fn = DeepClusterLoss(config.temperature)

    def on_train_start(self):
        self.dataset_size = len(self.trainer.train_dataloader.dataset)
        self.batch_size = self.trainer.config.trainer.batch_size

        self.kmeans = KMeans(
            nb_prototypes=self.config.nb_prototypes,
            nb_iters=self.config.kmeans_nb_iters,
            dataset_size=self.dataset_size,
        )

        self.assignments = -1 * torch.ones(
            (len(self.config.nb_prototypes), self.dataset_size),
            dtype=torch.long,
            device=self.trainer.device,
        )

        size_memory_per_process = (
            len(self.trainer.train_dataloader) * self.batch_size // get_world_size()
        )
        self.register_buffer(
            "local_memory_indexes",
            torch.zeros(size_memory_per_process)
            .long()
            .to(self.trainer.device, non_blocking=True),
        )
        self.register_buffer(
            "local_memory_embeddings",
            F.normalize(
                torch.randn(
                    2, size_memory_per_process, self.config.projector_output_dim
                ),
                dim=-1,
            ).to(self.trainer.device, non_blocking=True),
        )

    def forward(self, X: T, training: bool = False) -> Union[T, Tuple[T, T, T, T]]:
        if not training:
            return self.encoder(X)

        X_1 = X[:, 0, :]
        X_2 = X[:, 1, :]

        Z_1 = F.normalize(self.projector(self.encoder(X_1)), dim=-1)
        Z_2 = F.normalize(self.projector(self.encoder(X_2)), dim=-1)

        P_1 = torch.stack([layer(Z_1) for layer in self.prototypes], dim=1)
        P_2 = torch.stack([layer(Z_2) for layer in self.prototypes], dim=1)

        return Z_1, Z_2, P_1, P_2

    def get_learnable_params(self) -> Iterable[Dict[str, Any]]:
        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().get_learnable_params() + extra_learnable_params

    def on_train_epoch_start(self, epoch: int, max_epochs: int):
        if epoch > 0:
            self.assignments, centroids = self.kmeans.run(
                self.local_memory_indexes,
                self.local_memory_embeddings,
            )

            for layer, centroid in zip(self.prototypes, centroids):
                layer.weight.copy_(centroid)

    def train_step(
        self,
        Z: Tuple[T, T, T, T],
        step: int,
        step_rel: Optional[int] = None,
        indices: Optional[T] = None,
        labels: Optional[T] = None,
    ) -> T:
        Z_1, Z_2, P_1, P_2 = Z
        # P: (N, P, C)

        preds = torch.stack(
            (P_1.transpose(0, 1), P_2.transpose(0, 1)), dim=1
        )  # preds: (P, V, N, C)
        assignments = self.assignments[:, indices]

        loss = self.loss_fn(preds, assignments)

        # Store indexes and embeddings for the next clustering step
        start_idx = step_rel * self.batch_size // get_world_size()
        end_idx = (step_rel + 1) * self.batch_size // get_world_size()
        self.local_memory_indexes[start_idx:end_idx] = indices
        self.local_memory_embeddings[0, start_idx:end_idx] = Z_1.detach()
        self.local_memory_embeddings[1, start_idx:end_idx] = Z_2.detach()

        self.log_step_metrics(
            step,
            {
                "train/loss": loss,
            },
        )

        return loss
